from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
from torch.autograd import Variable
import util
import classifier_ZLAP
import model

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='AWA2', help='AWA')
parser.add_argument('--dataroot', default='../data/', help='path to dataset')
parser.add_argument('--matdataset', default=True, help='Data in matlab format')
parser.add_argument('--image_embedding', default='res101')
parser.add_argument('--class_embedding', default='att')
parser.add_argument('--syn_num', type=int, default=10, help='number features to generate per class')
parser.add_argument('--gzsl', action='store_true', default=True, help='enable generalized zero-shot learning')
parser.add_argument('--preprocessing', action='store_true', default=True, help='enbale MinMaxScaler on visual features')
parser.add_argument('--standardization', action='store_true', default=False)
parser.add_argument('--validation', action='store_true', default=False, help='enable cross validation mode')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batch_size', type=int, default=512, help='input batch size')
parser.add_argument('--resSize', type=int, default=2048, help='size of visual features')
parser.add_argument('--attSize', type=int, default=85, help='size of semantic features')
parser.add_argument('--nz', type=int, default=85, help='size of the latent z vector')
parser.add_argument('--latent_size', type=int, default=85, help='size of the latent z vector')
parser.add_argument('--ngh', type=int, default=4096, help='size of the hidden units in generator')
parser.add_argument('--ndh', type=int, default=4096, help='size of the hidden units in discriminator')
parser.add_argument('--nepoch', type=int, default=200, help='number of epochs to train for')
parser.add_argument('--critic_iter', type=int, default=5, help='critic iteration, following WGAN-GP')
parser.add_argument('--lambda1', type=float, default=10, help='gradient penalty regularizer, following WGAN-GP')
parser.add_argument('--cls_weight', type=float, default=0.0001, help='weight of the classification loss')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate to train GANs ')
parser.add_argument('--classifier_lr', type=float, default=0.0001, help='learning rate to train softmax classifier')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', default=True, help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--pretrain_classifier', default='', help="path to pretrain classifier (to continue training)")
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--manualSeed', type=int, default=None, help='manual seed')
parser.add_argument('--nclass_all', type=int, default=50, help='number of all classes')
parser.add_argument('--decoder_layer_sizes', type=list, default=[4096, 2048], help='number of all classes')
parser.add_argument('--netP', default='', help="path to netP (to continue training)")
parser.add_argument('--tem', type=float, default=0.04)
parser.add_argument('--prototype_layer_sizes', type=list, default=[1024,2048], help='')
parser.add_argument('--p_s', type=float, default=0.924)
parser.add_argument('--ratio', type=float, default=250)
opt = parser.parse_args()
print(opt)
opt.p_s=1/(1+1/opt.ratio)
print('p_s:',opt.p_s)

os.environ['CUDA_VISIBLE_DEVICES']='0'


# logger = util.Logger(opt.outname)

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)



# cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

# load data
data = util.DATA_LOADER(opt)
print("# of training samples: ", data.ntrain)

random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)
# initialize generator and discriminator
netG=model.netG(opt.decoder_layer_sizes, opt.latent_size, opt.attSize)
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
print(netG)
netD = model.netD(opt)
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
print(netD)

input_res = torch.FloatTensor(opt.batch_size, opt.resSize)
input_att = torch.FloatTensor(opt.batch_size, opt.attSize)
noise = torch.FloatTensor(opt.batch_size, opt.nz)
one = torch.tensor(1, dtype=torch.float)
mone = one * -1
input_label = torch.LongTensor(opt.batch_size)
crossentropyloss = nn.CrossEntropyLoss()
idx=torch.LongTensor(opt.batch_size)

if opt.cuda:
    netD=netD.cuda()
    netG=netG.cuda()
    input_res = input_res.cuda()
    noise, input_att = noise.cuda(), input_att.cuda()
    one = one.cuda()
    mone = mone.cuda()
    crossentropyloss=crossentropyloss.cuda()
    input_label = input_label.cuda()
    idx=idx.cuda()
def sample(batch_size=opt.batch_size):
    batch_feature, batch_label, batch_att,batch_idx = data.next_batch_change(batch_size)
    input_res.copy_(batch_feature)
    input_att.copy_(batch_att)
    input_label.copy_(util.map_label(batch_label, data.seenclasses))
    idx.copy_(batch_idx)
def gene_syn_feature(netG, classes, attribute, num):

    nclass = classes.size(0)
    syn_feature = torch.FloatTensor(nclass*num, opt.resSize)
    syn_label = torch.LongTensor(nclass*num)
    syn_att = torch.FloatTensor(num, opt.attSize)
    syn_noise = torch.FloatTensor(num, opt.nz)
    syn_att_out = torch.FloatTensor(nclass * num, opt.attSize)
    if opt.cuda:
        syn_att = syn_att.cuda()
        syn_noise = syn_noise.cuda()
    for i in range(nclass):
        iclass = classes[i]
        iclass_att = attribute[iclass]
        syn_att.copy_(iclass_att.repeat(num, 1).cuda())
        syn_noise.normal_(0, 1)
        with torch.no_grad():
            output = netG(Variable(syn_noise), Variable(syn_att))
        syn_feature.narrow(0, i*num, num).copy_(output.data.cpu())
        syn_label.narrow(0, i*num, num).fill_(iclass)
        syn_att_out.narrow(0, i * num, num).copy_(syn_att.data.cpu())
    return syn_feature, syn_label, syn_att_out

optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

def calc_gradient_penalty(netD, real_data, fake_data, input_att):

    alpha = torch.rand(opt.batch_size, 1)
    alpha = alpha.expand(real_data.size())
    if opt.cuda:
        alpha = alpha.cuda()

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    if opt.cuda:
        interpolates = interpolates.cuda()

    interpolates = Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates, Variable(input_att))

    ones = torch.ones(disc_interpolates.size())
    if opt.cuda:
        ones = ones.cuda()

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=ones,
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * opt.lambda1
    return gradient_penalty
def calc_gradient_penalty2(netD, real_data, fake_data, input_vis):

    alpha = torch.rand(opt.batch_size, 1)
    alpha = alpha.expand(real_data.size())
    if opt.cuda:
        alpha = alpha.cuda()

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    if opt.cuda:
        interpolates = interpolates.cuda()

    interpolates = Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(input_vis, interpolates)

    ones = torch.ones(disc_interpolates.size())
    if opt.cuda:
        ones = ones.cuda()

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=ones,
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * opt.lambda1
    return gradient_penalty


for epoch in range(opt.nepoch):
    for i in range(0, data.ntrain, opt.batch_size):
        ############################
        # (1) Update D network: optimize WGAN-GP objective, Equation (2)
        ###########################
        for p in netD.parameters():  # reset requires_grad
            p.requires_grad = True  # they are set to False below in netG and Encoder update
        for p in netG.parameters():  # reset requires_grad
            p.requires_grad = False # they are set to False below in netG and Encoder update
        for iter_d in range(opt.critic_iter):
            netD.zero_grad()
            sample(opt.batch_size)
            criticD_real = netD(input_res, input_att).mean()
            att_various =input_att
            noise.normal_(0, 1)
            fake = netG(noise, att_various).detach()
            criticD_fake = netD(fake, input_att).mean()
            # gradient penalty
            gradient_penalty = calc_gradient_penalty(netD, input_res, fake.data, input_att)

            W_D = (criticD_real - criticD_fake).item()
            D_cost = criticD_fake - criticD_real + gradient_penalty
            D_cost.backward()
            optimizerD.step()
            D_cost=D_cost.item()

        ############################
        # (2) Update G network: optimize WGAN-GP objective, Equation (2)
        ###########################
        for p in netD.parameters():  # reset requires_grad
            p.requires_grad = False  # avoid computation
        for p in netG.parameters():  # reset requires_grad
            p.requires_grad = True # they are set to False below in netG and Encoder update
        netG.zero_grad()
        fake = netG(noise, att_various)

        G_cost = -netD(fake, input_att).mean()

        cost_all=G_cost
        cost_all.backward()
        optimizerG.step()

    print('[%d/%d] Loss_D: %.4f Loss_G: %.4f W_d: %.4f'# rloss_fake: %.4f closs:%.4f -H_z:%.4f rloss_real%.4f'
          % (epoch, opt.nepoch, D_cost, G_cost.data, W_D))#, reg_loss_fake.data,closs.data,loss_H_z,reg_loss_real.data))

    syn_feature, syn_label,syn_att = gene_syn_feature(netG, data.unseenclasses, data.attribute,opt.syn_num)
    train_att_=data.attribute[data.train_label]
    train_X = torch.cat((data.train_feature, syn_feature), 0)
    train_Y = torch.cat((data.train_label, syn_label), 0)
    train_att=torch.cat((train_att_,syn_att))

    cls_gzsl = classifier_ZLAP.CLASSIFIER(opt.p_s,opt.prototype_layer_sizes, train_X,
                                               train_Y, data,
                                               opt.cuda, opt.classifier_lr, 0.5, 60,
                                               opt.batch_size,
                                               True,opt.tem)
    acu = cls_gzsl.acc_unseen.cpu().data.numpy()
    acs = cls_gzsl.acc_seen.cpu().data.numpy()
    ach = cls_gzsl.H.data.cpu().numpy()
    print('unseen=%.4f, seen=%.4f, h=%.4f' % (acu, acs, ach))
    netG.train()

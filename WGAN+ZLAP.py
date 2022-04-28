from __future__ import print_function
import argparse
import os
import random
import math
import torch
import torch.autograd as autograd
import torch.optim as optim
import util
import classifier_ZLAP
import model
os.environ['CUDA_VISIBLE_DEVICES']='0'

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', default='../data/', help='path to dataset')
parser.add_argument('--matdataset', default=True, help='Data in matlab format')
parser.add_argument('--image_embedding', default='res101')
parser.add_argument('--class_embedding', default='att')
parser.add_argument('--gzsl', action='store_true', default=True, help='enable generalized zero-shot learning')
parser.add_argument('--preprocessing', action='store_true', default=True, help='enbale MinMaxScaler on visual features')
parser.add_argument('--standardization', action='store_true', default=False)
parser.add_argument('--validation', action='store_true', default=False, help='enable cross validation mode')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batch_size', type=int, default=512, help='input batch size')
parser.add_argument('--resSize', type=int, default=2048, help='size of visual features')
parser.add_argument('--nepoch', type=int, default=2000, help='number of epochs to train for')
parser.add_argument('--critic_iter', type=int, default=5, help='critic iteration, following WGAN-GP')
parser.add_argument('--lambda1', type=float, default=10, help='gradient penalty regularizer, following WGAN-GP')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate to train GANs ')
parser.add_argument('--classifier_lr', type=float, default=0.0001, help='learning rate to train softmax classifier')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', default=True, help='enables cuda')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--netP', default='', help="path to netP (to continue training)")
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--manualSeed', type=int, default=70, help='manual seed')
parser.add_argument('--discriminator_layer_sizes', type=int, default=[4096], help='size of the hidden units in discriminator')
parser.add_argument('--generator_layer_sizes', type=list, default=[4096, 2048], help='size of the hidden and output units in generator')
parser.add_argument('--proto_layer_sizes', type=list, default=[1024,2048], help='size of the hidden and output units in prototype learner')
"""
For changing datasets
"""
parser.add_argument('--dataset', default='AWA2', help='set dataset')
parser.add_argument('--attSize', type=int, default=85, help='size of semantic features')
parser.add_argument('--nz', type=int, default=85, help='size of the latent z vector')

"""
Important hyperparameters in our paper
"""
parser.add_argument('--syn_num', type=int, default=10, help='number features to generate per unseen class')
parser.add_argument('--tem', type=float, default=0.04,help='temprature (Eq. 16)')
parser.add_argument('--ratio', type=float, default=1000,help='hyperparameter to control the seen-unseen prior (Sec. 4.4)')

opt = parser.parse_args()
print(opt)

# load data
data = util.DATA_LOADER(opt)
print("# of training samples: ", data.ntrain)

num_s=data.seenclasses.size(0)
num_u=data.unseenclasses.size(0)
num_class=num_s+num_u

"""
class prior
"""
log_p_y = torch.zeros(num_class).cuda()#class prior
p_y_seen = torch.zeros(num_s).cuda()#conditional class prior on seen class (near Eq. 14)
p_y_unseen = torch.ones(num_u).cuda() / num_u#conditional class prior on unseen class (near Eq. 14)
for i in range(p_y_seen.size(0)):
    iclass = data.seenclasses[i]
    index = data.train_label == iclass
    p_y_seen[i] = index.sum().float().cuda()
p_y_seen /= p_y_seen.sum()
log_p_y[data.seenclasses] = p_y_seen.log()
log_p_y[data.unseenclasses] = p_y_unseen.log()
"""
seen-unseen prior
"""
log_p0_Y = torch.zeros(num_class).cuda()#seen-unsee prior (Eq. 11)
p0_s=1/(1+1/opt.ratio)
p0_u = (1 - p0_s)
log_p0_Y[data.seenclasses] = math.log(p0_s)
log_p0_Y[data.unseenclasses] = math.log(p0_u)

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")
if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)

random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)
# initialize generator and discriminator
netG=model.netG(opt.generator_layer_sizes, opt.nz, opt.attSize)
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
print(netG)
netD = model.netD(opt.discriminator_layer_sizes,opt.resSize,opt.attSize)
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
print(netD)

input_res = torch.FloatTensor(opt.batch_size, opt.resSize)
input_label = torch.LongTensor(opt.batch_size)
noise = torch.FloatTensor(opt.batch_size, opt.nz)

if opt.cuda:
    netD=netD.cuda()
    netG=netG.cuda()
    input_res = input_res.cuda()
    noise = noise.cuda()
    input_label = input_label.cuda()

def sample(batch_size=opt.batch_size):
    batch_feature, batch_label= data.next_batch(batch_size)
    input_res.copy_(batch_feature)
    input_label.copy_(batch_label)


def generate_syn_feature(netG, classes, num):
    nclass = classes.size(0)
    syn_feature = torch.FloatTensor(nclass * num, opt.resSize)
    syn_label = torch.LongTensor(nclass * num)
    syn_att = torch.FloatTensor(num, opt.attSize)
    syn_noise = torch.FloatTensor(num, opt.nz)
    if opt.cuda:
        syn_att = syn_att.cuda()
        syn_noise = syn_noise.cuda()

    for i in range(nclass):
        iclass = classes[i]
        iclass_att = data.attribute[iclass]
        syn_att.copy_(iclass_att.repeat(num, 1))
        syn_noise.normal_(0, 1)
        with torch.no_grad():
            output = netG(syn_noise, syn_att)
        syn_feature.narrow(0, i * num, num).copy_(output.data.cpu())
        syn_label.narrow(0, i * num, num).fill_(iclass)

    return syn_feature, syn_label

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
    interpolates = interpolates.requires_grad_(True)
    disc_interpolates = netD(interpolates, input_att)
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
        # (1) Update D network: optimize WGAN-GP objective
        ###########################
        for p in netD.parameters():  # reset requires_grad
            p.requires_grad = True  # they are set to False below in netG update
        for p in netG.parameters():  # reset requires_grad
            p.requires_grad = False # they are set to False below in netG update
        for iter_d in range(opt.critic_iter):
            netD.zero_grad()
            sample(opt.batch_size)
            input_att=data.attribute[input_label].cuda()
            criticD_real = netD(input_res, input_att).mean()
            noise.normal_(0, 1)
            fake = netG(noise, input_att).detach()
            criticD_fake = netD(fake, input_att).mean()
            # gradient penalty
            gradient_penalty = calc_gradient_penalty(netD, input_res, fake.data, input_att)

            W_D = (criticD_real - criticD_fake).item()
            D_cost = criticD_fake - criticD_real + gradient_penalty
            D_cost.backward()
            optimizerD.step()
            D_cost=D_cost.item()

        ############################
        # (2) Update G network: optimize WGAN-GP objective
        ###########################
        for p in netD.parameters():  # reset requires_grad
            p.requires_grad = False  # avoid computation
        for p in netG.parameters():  # reset requires_grad
            p.requires_grad = True # they are set to False below in netG update
        netG.zero_grad()
        fake = netG(noise, input_att)

        G_cost = -netD(fake, input_att).mean()

        cost_all=G_cost
        cost_all.backward()
        optimizerG.step()

    print('[%d/%d] Loss_D: %.4f Loss_G: %.4f W_d: %.4f'% (epoch, opt.nepoch, D_cost, G_cost.data, W_D))

    syn_feature, syn_label = generate_syn_feature(netG, data.unseenclasses,opt.syn_num)
    train_X = torch.cat((data.train_feature, syn_feature), 0)
    train_Y = torch.cat((data.train_label, syn_label), 0)

    cls_gzsl = classifier_ZLAP.CLASSIFIER(log_p_y,log_p0_Y,opt.proto_layer_sizes, train_X,
                                               train_Y, data,
                                               opt.cuda, opt.classifier_lr, 0.5, 60,
                                               opt.batch_size,
                                               True,opt.tem)
    acu = cls_gzsl.acc_unseen.cpu().data.numpy()
    acs = cls_gzsl.acc_seen.cpu().data.numpy()
    ach = cls_gzsl.H.data.cpu().numpy()
    print('unseen=%.4f, seen=%.4f, h=%.4f' % (acu, acs, ach))
    netG.train()

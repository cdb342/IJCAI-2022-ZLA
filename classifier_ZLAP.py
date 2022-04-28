import torch
import torch.nn.functional as F
import torch.optim as optim
import model
class CLASSIFIER:
    # train_Y is interger
    def __init__(self, log_p_y,log_p0_Y, prototype_layer_sizes, _train_X, _train_Y, data_loader, _cuda, _lr=0.0001,
                 _beta1=0.5, _nepoch=20,
                 _batch_size=100, generalized=True, tem=0.04):

        self.train_X = _train_X
        self.train_Y = _train_Y
        self.test_seen_feature = data_loader.test_seen_feature
        self.test_seen_label = data_loader.test_seen_label
        self.test_unseen_feature = data_loader.test_unseen_feature
        self.test_unseen_label = data_loader.test_unseen_label
        self.seenclasses = data_loader.seenclasses.cuda()
        self.unseenclasses = data_loader.unseenclasses.cuda()
        self.att = data_loader.attribute
        self.batch_size = _batch_size
        self.nepoch = _nepoch
        self.input_dim = _train_X.size(1)
        self.cuda = _cuda
        self.netP = model.netP(prototype_layer_sizes, self.att.size(-1))
        self.optimizerP = optim.Adam(self.netP.parameters(), _lr, betas=(_beta1, 0.999),weight_decay=0.0001)
        self.input = torch.FloatTensor(_batch_size, self.input_dim)
        self.label = torch.LongTensor(_batch_size)
        self.lr = _lr
        self.beta1 = _beta1
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizerP, gamma=0.5, step_size=30)
        self.tem = tem
        self.log_p_y=log_p_y
        self.log_p0_Y=log_p0_Y
        if self.cuda:
            self.netP = self.netP.cuda()
            self.input = self.input.cuda()
            self.label = self.label.cuda()
            self.att=self.att.cuda()

        self.index_in_epoch = 0
        self.epochs_completed = 0
        self.ntrain = self.train_X.size()[0]
        if generalized:
            self.acc_seen, self.acc_unseen, self.H,self.best_epoch = self.fit_ZLA()

    def fit_ZLA(self):
        best_H = torch.zeros(1).cuda()
        best_seen = torch.zeros(1).cuda()
        best_unseen = torch.zeros(1).cuda()
        best_epoch = torch.zeros(1)
        for epoch in range(self.nepoch):
            self.netP.train()
            for i in range(0, self.ntrain, self.batch_size):
                self.netP.zero_grad()
                batch_input, batch_label = self.next_batch(self.batch_size)
                batch_input = F.normalize(batch_input, dim=1).cuda()
                batch_label = batch_label.cuda()
                proto=F.normalize(self.netP(self.att),dim=-1)
                logits = batch_input@proto.t()/self.tem
                """
                Zero-Shot Logit Adjustment
                """
                logits = logits+self.log_p_y+self.log_p0_Y

                loss = F.cross_entropy(logits,batch_label)
                loss.backward()
                self.optimizerP.step()
            self.scheduler.step()
            self.netP.eval()
            acc_seen = self.val_gzsl(self.test_seen_feature, self.test_seen_label, self.seenclasses)
            acc_unseen = self.val_gzsl(self.test_unseen_feature, self.test_unseen_label,
                                       self.unseenclasses)
            H = 2 * acc_seen * acc_unseen / (acc_seen + acc_unseen)
            if H > best_H:
                best_seen = acc_seen
                best_unseen = acc_unseen
                best_H = H
                best_epoch=epoch
        return best_seen, best_unseen, best_H,best_epoch

    def val_gzsl(self, test_X, test_label, target_classes):
        start = 0
        ntest = test_X.size()[0]
        predicted_label = torch.LongTensor(test_label.size()).cuda()
        test_label = test_label.cuda()
        target_classes = target_classes.cuda()
        att_proto=self.att
        for i in range(0, ntest, self.batch_size):
            end = min(ntest, start + self.batch_size)
            with torch.no_grad():
                test_batch = F.normalize(test_X[start:end], dim=-1).cuda()
                proto = F.normalize(self.netP(att_proto), dim=-1)
                output = test_batch@proto.t()
            predicted_label[start:end] = torch.max(output.data, 1)[1]
            start = end
        acc = self.compute_per_class_acc_gzsl(test_label, predicted_label, target_classes)
        return acc

    def compute_per_class_acc_gzsl(self, test_label, predicted_label, target_classes):
        acc_per_class = 0
        for i in target_classes:
            idx = (test_label == i)
            acc_per_class += torch.sum(test_label[idx] == predicted_label[idx]).float() / torch.sum(idx).float()
        acc_per_class /= target_classes.size(0)
        return acc_per_class

    def next_batch(self, batch_size):
        start = self.index_in_epoch
        # shuffle the data at the first epoch
        if self.epochs_completed == 0 and start == 0:
            perm = torch.randperm(self.ntrain)
            self.train_X = self.train_X[perm]
            self.train_Y = self.train_Y[perm]
        # the last batch
        if start + batch_size > self.ntrain:
            self.epochs_completed += 1
            rest_num_examples = self.ntrain - start
            if rest_num_examples > 0:
                X_rest_part = self.train_X[start:self.ntrain]
                Y_rest_part = self.train_Y[start:self.ntrain]
            # shuffle the data
            perm = torch.randperm(self.ntrain)
            self.train_X = self.train_X[perm]
            self.train_Y = self.train_Y[perm]
            # start next epoch
            start = 0
            self.index_in_epoch = batch_size - rest_num_examples
            end = self.index_in_epoch
            X_new_part = self.train_X[start:end]
            Y_new_part = self.train_Y[start:end]
            # print(start, end)
            if rest_num_examples > 0:
                return torch.cat((X_rest_part, X_new_part), 0), torch.cat((Y_rest_part, Y_new_part), 0)
            else:
                return X_new_part, Y_new_part
        else:
            self.index_in_epoch += batch_size
            end = self.index_in_epoch
            # print(start, end)
            # from index start to index end-1
            return self.train_X[start:end], self.train_Y[start:end]
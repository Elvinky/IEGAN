import time, itertools
from dataset import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
from networks import *
from utils import *
from glob import glob
# import torch.utils.tensorboard as tensorboardX
from thop import profile
from thop import clever_format

class IEGAN(object) :
    def __init__(self, args):
        self.light = args.light
        self.model_name = 'IEGAN'

        self.result_dir = args.result_dir
        self.dataset = args.dataset

        self.iteration = args.iteration
        self.decay_flag = args.decay_flag

        self.batch_size = args.batch_size
        self.print_freq = args.print_freq
        self.save_freq = args.save_freq

        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.ch = args.ch

        """ Weight """
        self.adv_weight = args.adv_weight
        self.cycle_weight = args.cycle_weight
        self.recon_weight = args.recon_weight

        """ Generator """
        self.n_res = args.n_res

        """ Discriminator """
        self.n_dis = args.n_dis

        self.img_size = args.img_size
        self.img_ch = args.img_ch

        self.device = args.device
        self.benchmark_flag = args.benchmark_flag
        self.resume = args.resume

        self.start_iter = 1

        self.fid = 1000
        self.fid_A = 1000
        self.fid_B = 1000
        if torch.backends.cudnn.enabled and self.benchmark_flag:
            print('set benchmark !')
            torch.backends.cudnn.benchmark = True

        print()

        print("##### Information #####")
        print("# light : ", self.light)
        print("# dataset : ", self.dataset)
        print("# batch_size : ", self.batch_size)
        print("# iteration per epoch : ", self.iteration)
        print("# the size of image : ", self.img_size)
        print("# the size of image channel : ", self.img_ch)
        print("# base channel number per layer : ", self.ch)

        print()

        print("##### Generator #####")
        print("# residual blocks : ", self.n_res)

        print()

        print("##### Discriminator #####")
        print("# discriminator layers : ", self.n_dis)

        print()

        print("##### Weight #####")
        print("# adv_weight : ", self.adv_weight)
        print("# cycle_weight : ", self.cycle_weight)
        print("# recon_weight : ", self.recon_weight)

    ##################################################################################
    # Model
    ##################################################################################

    def build_model(self):
        """ DataLoader """
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize((self.img_size + 30, self.img_size+30)),
            transforms.RandomCrop(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        test_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

        self.trainA = ImageFolder(os.path.join('dataset', self.dataset, 'trainA'), train_transform)
        self.trainB = ImageFolder(os.path.join('dataset', self.dataset, 'trainB'), train_transform)
        self.testA = ImageFolder(os.path.join('dataset', self.dataset, 'testA'), test_transform)
        self.testB = ImageFolder(os.path.join('dataset', self.dataset, 'testB'), test_transform)
        self.trainA_loader = DataLoader(self.trainA, batch_size=self.batch_size, shuffle=True,pin_memory=True)
        self.trainB_loader = DataLoader(self.trainB, batch_size=self.batch_size, shuffle=True,pin_memory=True)
        self.testA_loader = DataLoader(self.testA, batch_size=1, shuffle=False,pin_memory=True)
        self.testB_loader = DataLoader(self.testB, batch_size=1, shuffle=False,pin_memory=True)

        """ Define Generator, Discriminator """
        self.gen2B = ResnetGenerator(input_nc=self.img_ch, output_nc=self.img_ch, ngf=self.ch, n_blocks=self.n_res, img_size=self.img_size, light=self.light).to(self.device)
        self.gen2A = ResnetGenerator(input_nc=self.img_ch, output_nc=self.img_ch, ngf=self.ch, n_blocks=self.n_res, img_size=self.img_size, light=self.light).to(self.device)
        self.disA = Discriminator(input_nc=self.img_ch, ndf=self.ch, n_layers=self.n_dis).to(self.device)
        self.disB = Discriminator(input_nc=self.img_ch, ndf=self.ch, n_layers=self.n_dis).to(self.device)
        self.encA = Encoder(3).to(self.device)
        self.encB = Encoder(3).to(self.device)

        """ Define Loss """
        self.L1_loss = nn.L1Loss().to(self.device)
        self.MSE_loss = nn.MSELoss().to(self.device)

        """ Trainer """ 
        self.G_optim = torch.optim.Adam(itertools.chain(self.gen2B.parameters(), self.gen2A.parameters()), lr=self.lr, betas=(0.5, 0.999), weight_decay=self.weight_decay)
        self.D_optim = torch.optim.Adam(itertools.chain(self.disA.parameters(), self.disB.parameters()), lr=self.lr, betas=(0.5, 0.999), weight_decay=self.weight_decay)
        self.E_optim = torch.optim.Adam(itertools.chain(self.encA.parameters(), self.encB.parameters()), lr=self.lr, betas=(0.5, 0.999), weight_decay=self.weight_decay)
        

    def train(self):
        # writer = tensorboardX.SummaryWriter(os.path.join(self.result_dir, self.dataset, 'summaries/Allothers'))
        self.gen2B.train(), self.gen2A.train(), self.disA.train(), self.disB.train(), self.encA.train(), self.encB.train()

        self.start_iter = 1
        if self.resume:
            params = torch.load('/home/kye18/temp/checkpoints/_params_3100000.pt')
            self.encA.load_state_dict(params, False)
            self.encB.load_state_dict(params, False)
            print("ok")
          

        # training loop
        testnum = 4
        for step in range(1, self.start_iter):
            if step % self.print_freq == 0:
                for _ in range(testnum):
                    try:
                        real_A, _ = testA_iter.next()
                    except:
                        testA_iter = iter(self.testA_loader)
                        real_A, _ = testA_iter.next()

                    try:
                        real_B, _ = testB_iter.next()
                    except:
                        testB_iter = iter(self.testB_loader)
                        real_B, _ = testB_iter.next()

        print("self.start_iter",self.start_iter)
        print('training start !')
        start_time = time.time()
        for step in range(self.start_iter, self.iteration + 1):
            if self.decay_flag and step > (self.iteration // 2):
                self.G_optim.param_groups[0]['lr'] -= (self.lr / (self.iteration // 2))
                self.D_optim.param_groups[0]['lr'] -= (self.lr / (self.iteration // 2))
                self.E_optim.param_groups[0]['lr'] -= (self.lr / (self.iteration // 2))
            try:
                real_A, _ = trainA_iter.next()
            except:
                trainA_iter = iter(self.trainA_loader)
                real_A, _ = trainA_iter.next()

            try:
                real_B, _ = trainB_iter.next()
            except:
                trainB_iter = iter(self.trainB_loader)
                real_B, _ = trainB_iter.next()

            real_A, real_B = real_A.to(self.device), real_B.to(self.device)
            
            # Update E
            self.E_optim.zero_grad()
            _, _, _, _, ae_A = self.encA(real_A)
            _, _, _, _, ae_B = self.encB(real_B)

            E_loss_A = self.L1_loss(ae_A, real_A)
            E_loss_B = self.L1_loss(ae_B, real_B)
            E_loss = 10* (E_loss_A + E_loss_B)
            
            E_loss.backward()
            self.E_optim.step()

            # Update D
            self.D_optim.zero_grad()
            
            real_A_E1, real_A_E2, real_A_E3, real_A_E4, _ = self.encA(real_A)
            real_B_E1, real_B_E2, real_B_E3, real_B_E4, _ = self.encB(real_B)
            
            real_LA_logit,real_GA_logit, real_A_cam_logit, _ = self.disA(real_A_E1)
            real_LB_logit,real_GB_logit, real_B_cam_logit, _ = self.disB(real_B_E1)

            fake_A2B = self.gen2B(real_A_E1, real_A_E2, real_A_E3, real_A_E4)
            fake_B2A = self.gen2A(real_B_E1, real_B_E2, real_B_E3, real_B_E4)

            fake_B2A = fake_B2A.detach()
            fake_A2B = fake_A2B.detach()
            
            fake_A_E1, fake_A_E2, fake_A_E3, fake_A_E4, _ = self.encA(fake_B2A)
            fake_B_E1, fake_B_E2, fake_B_E3, fake_B_E4, _ = self.encB(fake_A2B)

            fake_LA_logit, fake_GA_logit, fake_A_cam_logit, _ = self.disA(fake_A_E1)
            fake_LB_logit, fake_GB_logit, fake_B_cam_logit, _ = self.disB(fake_B_E1)


            D_ad_loss_GA = self.MSE_loss(real_GA_logit, torch.ones_like(real_GA_logit).to(self.device)) + self.MSE_loss(fake_GA_logit, torch.zeros_like(fake_GA_logit).to(self.device))
            D_ad_loss_LA = self.MSE_loss(real_LA_logit, torch.ones_like(real_LA_logit).to(self.device)) + self.MSE_loss(fake_LA_logit, torch.zeros_like(fake_LA_logit).to(self.device))
            D_ad_loss_GB = self.MSE_loss(real_GB_logit, torch.ones_like(real_GB_logit).to(self.device)) + self.MSE_loss(fake_GB_logit, torch.zeros_like(fake_GB_logit).to(self.device))
            D_ad_loss_LB = self.MSE_loss(real_LB_logit, torch.ones_like(real_LB_logit).to(self.device)) + self.MSE_loss(fake_LB_logit, torch.zeros_like(fake_LB_logit).to(self.device))            
            D_ad_cam_loss_A = self.MSE_loss(real_A_cam_logit, torch.ones_like(real_A_cam_logit).to(self.device)) + self.MSE_loss(fake_A_cam_logit, torch.zeros_like(fake_A_cam_logit).to(self.device))
            D_ad_cam_loss_B = self.MSE_loss(real_B_cam_logit, torch.ones_like(real_B_cam_logit).to(self.device)) + self.MSE_loss(fake_B_cam_logit, torch.zeros_like(fake_B_cam_logit).to(self.device))

            D_loss_A = self.adv_weight * (D_ad_loss_GA + D_ad_cam_loss_A + D_ad_loss_LA)
            D_loss_B = self.adv_weight * (D_ad_loss_GB + D_ad_cam_loss_B + D_ad_loss_LB)

            Discriminator_loss = D_loss_A + D_loss_B
            Discriminator_loss.backward()
            self.D_optim.step()
            # writer.add_scalar('D/%s' % 'loss_A', D_loss_A.data.cpu().numpy(), global_step=step)  
            # writer.add_scalar('D/%s' % 'loss_B', D_loss_B.data.cpu().numpy(), global_step=step)  

            # Update G
            self.G_optim.zero_grad()

            real_A_E1, real_A_E2, real_A_E3, real_A_E4, _ = self.encA(real_A)
            real_B_E1, real_B_E2, real_B_E3, real_B_E4, _ = self.encB(real_B)
            
            fake_A2B = self.gen2B(real_A_E1, real_A_E2, real_A_E3, real_A_E4)
            fake_B2A = self.gen2A(real_B_E1, real_B_E2, real_B_E3, real_B_E4)

            fake_A_E1, fake_A_E2, fake_A_E3, fake_A_E4, _ = self.encA(fake_B2A)
            fake_B_E1, fake_B_E2, fake_B_E3, fake_B_E4, _ = self.encB(fake_A2B)
            
            fake_LA_logit, fake_GA_logit, fake_A_cam_logit, _ = self.disA(fake_A_E1)
            fake_LB_logit, fake_GB_logit, fake_B_cam_logit, _ = self.disB(fake_B_E1)
            
            fake_B2A2B = self.gen2B(fake_A_E1, fake_A_E2, fake_A_E3, fake_A_E4)
            fake_A2B2A = self.gen2A(fake_B_E1, fake_B_E2, fake_B_E3, fake_B_E4)

            G_ad_loss_GA = self.MSE_loss(fake_GA_logit, torch.ones_like(fake_GA_logit).to(self.device))
            G_ad_loss_LA = self.MSE_loss(fake_LA_logit, torch.ones_like(fake_LA_logit).to(self.device))
            G_ad_loss_GB = self.MSE_loss(fake_GB_logit, torch.ones_like(fake_GB_logit).to(self.device))
            G_ad_loss_LB = self.MSE_loss(fake_LB_logit, torch.ones_like(fake_LB_logit).to(self.device))

            G_ad_cam_loss_A = self.MSE_loss(fake_A_cam_logit, torch.ones_like(fake_A_cam_logit).to(self.device))
            G_ad_cam_loss_B = self.MSE_loss(fake_B_cam_logit, torch.ones_like(fake_B_cam_logit).to(self.device))

            G_cycle_loss_A = self.L1_loss(fake_A2B2A, real_A)
            G_cycle_loss_B = self.L1_loss(fake_B2A2B, real_B)

            fake_A2A = self.gen2A(real_A_E1, real_A_E2, real_A_E3, real_A_E4)
            fake_B2B = self.gen2B(real_B_E1, real_B_E2, real_B_E3, real_B_E4)

            G_recon_loss_A = self.L1_loss(fake_A2A, real_A)
            G_recon_loss_B = self.L1_loss(fake_B2B, real_B)


            G_loss_A = self.adv_weight * (G_ad_loss_GA + G_ad_cam_loss_A + G_ad_loss_LA ) + self.cycle_weight * G_cycle_loss_A + self.recon_weight * G_recon_loss_A
            G_loss_B = self.adv_weight * (G_ad_loss_GB + G_ad_cam_loss_B + G_ad_loss_LB ) + self.cycle_weight * G_cycle_loss_B + self.recon_weight * G_recon_loss_B

            Generator_loss = G_loss_A + G_loss_B
            Generator_loss.backward()
            self.G_optim.step()
            # writer.add_scalar('G/%s' % 'loss_A', G_loss_A.data.cpu().numpy(), global_step=step)  
            # writer.add_scalar('G/%s' % 'loss_B', G_loss_B.data.cpu().numpy(), global_step=step)  

            print("[%5d/%5d] time: %4.4f d_loss: %.8f, g_loss: %.8f, e_loss: %.8f" % (step, self.iteration, time.time() - start_time, Discriminator_loss, Generator_loss, E_loss))

            # for name, param in self.gen2B.named_parameters():
            #     writer.add_histogram(name + "_gen2B", param.data.cpu().numpy(), global_step=step)

            # for name, param in self.gen2A.named_parameters():
            #     writer.add_histogram(name + "_gen2A", param.data.cpu().numpy(), global_step=step)

            # for name, param in self.disA.named_parameters():
            #     writer.add_histogram(name + "_disA", param.data.cpu().numpy(), global_step=step)

            # for name, param in self.disB.named_parameters():
            #     writer.add_histogram(name + "_disB", param.data.cpu().numpy(), global_step=step)

            
            if step % self.save_freq == 0:
                self.save(os.path.join(self.result_dir, self.dataset, 'model'), step)
                
    def save(self, dir, step):
        params = {}
        params['gen2B'] = self.gen2B.state_dict()
        params['gen2A'] = self.gen2A.state_dict()
        params['disA'] = self.disA.state_dict()
        params['disB'] = self.disB.state_dict()
        params['encA'] = self.encA.state_dict()
        params['encB'] = self.encB.state_dict()
        params['D_optimizer'] = self.D_optim.state_dict()
        params['G_optimizer'] = self.G_optim.state_dict()
        params['E_optimizer'] = self.E_optim.state_dict()
        params['start_iter'] = step
        torch.save(params, os.path.join(dir, self.dataset + '_params_%07d.pt' % step))


    def save_path(self, path_g,step):
        params = {}
        params['gen2B'] = self.gen2B.state_dict()
        params['gen2A'] = self.gen2A.state_dict()
        params['disA'] = self.disA.state_dict()
        params['disB'] = self.disB.state_dict()
        params['encA'] = self.encA.state_dict()
        params['encB'] = self.encB.state_dict()
        params['D_optimizer'] = self.D_optim.state_dict()
        params['G_optimizer'] = self.G_optim.state_dict()
        params['E_optimizer'] = self.E_optim.state_dict()
        params['start_iter'] = step
        torch.save(params, os.path.join(self.result_dir, self.dataset + path_g))

    def load(self):
        params = torch.load(os.path.join(self.result_dir, self.dataset, 'model', self.dataset + '_params_0100000.pt'))
        self.gen2B.load_state_dict(params['gen2B'])
        self.gen2A.load_state_dict(params['gen2A'])
        self.disA.load_state_dict(params['disA'])
        self.disB.load_state_dict(params['disB'])
        self.encA.load_state_dict(params['encA'])
        self.encB.load_state_dict(params['encB'])
        self.D_optim.load_state_dict(params['D_optimizer'])
        self.G_optim.load_state_dict(params['G_optimizer'])
        self.E_optim.load_state_dict(params['E_optimizer'])
        self.start_iter = params['start_iter']

    def test(self):
        self.load()
        print(self.start_iter)

        self.gen2B.eval(), self.gen2A.eval(), self.disA.eval(),self.disB.eval(),self.encA.eval(),self.encB.eval()
        for n, (real_A, real_A_path) in enumerate(self.testA_loader):
            real_A = real_A.to(self.device)
            real_A_E1, real_A_E2, real_A_E3, real_A_E4, _ = self.encA(real_A)
            fake_A2B = self.gen2B(real_A_E1, real_A_E2, real_A_E3, real_A_E4)

            A2B = RGB2BGR(tensor2numpy(denorm(fake_A2B[0])))
            print(real_A_path[0])
            cv2.imwrite(os.path.join(self.result_dir, self.dataset, 'fakeB', real_A_path[0].split('/')[-1]), A2B * 255.0)

        for n, (real_B, real_B_path) in enumerate(self.testB_loader):
            real_B = real_B.to(self.device)
            real_B_E1, real_B_E2, real_B_E3, real_B_E4, _ = self.encB(real_B)
            fake_B2A = self.gen2A(real_B_E1, real_B_E2, real_B_E3, real_B_E4)

            B2A = RGB2BGR(tensor2numpy(denorm(fake_B2A[0])))
            print(real_B_path[0])
            cv2.imwrite(os.path.join(self.result_dir, self.dataset, 'fakeA', real_B_path[0].split('/')[-1]), B2A * 255.0)

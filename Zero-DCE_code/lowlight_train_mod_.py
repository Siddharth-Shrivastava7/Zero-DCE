import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time
import dataloader2
import model
import Myloss
import numpy as np
from torchvision import transforms
# import wandb
from tensorboardX import SummaryWriter
from time import time
from torch.nn.parallel import DistributedDataParallel as DDP

writer = SummaryWriter()

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def train(config):

	# wandb.init(project='zerodce')

	# wandb.config.epochs = 1000
	# wandb.config.batch_size = 8
	# wandb.config.learning_rate = config.lr
	# wandb.config.architecture = "zerodce"

	# os.environ['CUDA_VISIBLE_DEVICES']='0'

	DCE_net = nn.DataParallel(model.enhance_net_nopool()).cuda()
	# DCE_net = DDP(model.enhance_net_nopool()).cuda()

	# DCE_net = model.enhance_net_nopool().cuda()

	DCE_net.apply(weights_init)
	if config.load_pretrain == True:
	    DCE_net.load_state_dict(torch.load(config.pretrain_dir))
	train_dataset = dataloader2.lowlight_loader(config.lowlight_images_path)		
	val_dataset = dataloader2.lowlight_loader(config.val_images_path)    #have to give loc for val_images_path
	
 
	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True)
	val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.val_batch_size,shuffle=True, num_workers=config.num_workers, pin_memory=True)

	L_color = Myloss.L_color()
	L_spa = Myloss.L_spa()

	L_exp = Myloss.L_exp(16,0.6)
	L_TV = Myloss.L_TV()

	optimizer = torch.optim.Adam(DCE_net.parameters(), lr=config.lr, weight_decay=config.weight_decay)

	DCE_net.train()

	# wandb.watch(DCE_net)

	best_val_epoch_loss = float('inf')

	count = 0 

	for epoch in range(config.num_epochs):
		tic = time()
		train_epoch_loss = 0
		valid_epoch_loss = 0

		for iteration, img_lowlight in enumerate(train_loader):

			img_lowlight = img_lowlight.cuda()

			enhanced_image_1,enhanced_image,A  = DCE_net(img_lowlight)

			Loss_TV = 200*L_TV(A)
			
			loss_spa = torch.mean(L_spa(enhanced_image, img_lowlight))

			loss_col = 5*torch.mean(L_color(enhanced_image))

			loss_exp = 10*torch.mean(L_exp(enhanced_image))
			
			# best_loss
			loss =  Loss_TV + loss_spa + loss_col + loss_exp
			#
			
			optimizer.zero_grad()
			loss.backward()
			torch.nn.utils.clip_grad_norm(DCE_net.parameters(),config.grad_clip_norm)
			optimizer.step()

			train_epoch_loss += loss.item()

			# if ((iteration+1) % config.display_iter) == 0:
			# 	print("trainLoss at iteration", iteration+1, ":", loss.item())
			
			# if ((iteration+1) % config.snapshot_iter) == 0:

			# 	torch.save(DCE_net.state_dict(), config.snapshots_folder + "Epoch" + str(epoch) + '.pth') 
		
		train_epoch_loss /= len(iter(train_loader))

		print('in Training')
		print( 'time:', int(time() - tic))
		print('epoch:', epoch, 'train_loss:', train_epoch_loss)

		print('-------------------------------------------------------')
		
		for iteration, vimg_lowlight in enumerate(val_loader):
			vimg_lowlight = vimg_lowlight.cuda()

			venhanced_image_1,venhanced_image,vA  = DCE_net(vimg_lowlight)

			vLoss_TV = 200*L_TV(vA)
			
			vloss_spa = torch.mean(L_spa(venhanced_image, vimg_lowlight))

			vloss_col = 5*torch.mean(L_color(venhanced_image))

			vloss_exp = 10*torch.mean(L_exp(venhanced_image))
			
			# best_loss
			valloss =  vLoss_TV + vloss_spa + vloss_col + vloss_exp

			valid_epoch_loss += valloss.item()

			# if (valloss < best_loss):
			# 	best_loss = valloss.item()
			# 	print('best_loss:', best_loss)
			# 	print('Model_updated')
			# 	torch.save(DCE_net.state_dict(), config.snapshots_folder + "Epoch" + str(epoch) + '.pth')
		
		valid_epoch_loss /= len(iter(val_loader))

		print('in Validation')
		print('epoch:', epoch, 'valid_loss:', valid_epoch_loss)

		writer.add_scalars('Epoch_loss',{'train_epoch_loss': train_epoch_loss,'valid_epoch_loss':valid_epoch_loss},epoch)

		# writer.add_scalar('train_epoch_loss', train_epoch_loss, epoch)
		# writer.add_scalar('valid_epoch_loss', valid_epoch_loss, epoch)

		# wandb.log({"train_loss": loss.item(), "val_loss": valloss.item(), "best_val_loss": best_loss})

		if(valid_epoch_loss < best_val_epoch_loss):
			best_val_epoch_loss = valid_epoch_loss
			count = 0
			print('best_val_epoch_loss: ', best_val_epoch_loss)
			print("MODEL UPDATED")
			torch.save(DCE_net.state_dict(), config.snapshots_folder + "Epoch" + str(epoch) + '.pth')

		# if(valid_epoch_loss > best_val_epoch_loss):
		# 	count += 1 
		# 	if(count==30):
		# 		print('Early Stopping at 30 patience')
		# 		break

		print('*******************************************************************')
	
	print(count)
	writer.export_scalars_to_json("./all_scalars.json")
	writer.close()

if __name__ == "__main__":

	parser = argparse.ArgumentParser()

	# Input Parameters
	parser.add_argument('--lowlight_images_path', type=str, default="data/train_data/")
	parser.add_argument('--val_images_path', type=str, default="data/val_data/")

	parser.add_argument('--lr', type=float, default=0.0001)
	parser.add_argument('--weight_decay', type=float, default=0.0001)
	parser.add_argument('--grad_clip_norm', type=float, default=0.1)
	parser.add_argument('--num_epochs', type=int, default=200)
	parser.add_argument('--train_batch_size', type=int, default=8)
	parser.add_argument('--val_batch_size', type=int, default=4)
	parser.add_argument('--num_workers', type=int, default=8)  # num_work = 4 * num_gpu
	parser.add_argument('--display_iter', type=int, default=10)
	parser.add_argument('--snapshot_iter', type=int, default=10)
	parser.add_argument('--snapshots_folder', type=str, default="snapshots/")
	parser.add_argument('--load_pretrain', type=bool, default= False)
	parser.add_argument('--pretrain_dir', type=str, default= "snapshots/Epoch99.pth")

	config = parser.parse_args()

	if not os.path.exists(config.snapshots_folder):
		os.mkdir(config.snapshots_folder)


	train(config)








	

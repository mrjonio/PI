import torch
import time
import os
from torch.autograd import Variable
import skimage.io as io
import data_loader_origin as dto
from torch.utils.data import Dataset, DataLoader
from data_utils import data_mean_value
import numpy as np
from torchvision import utils
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import normalized_root_mse as rmse
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


#net             = 'RED-CNN'
net             = 'GOOGLENET'

tipo = 1

input_dir = "/home/calves/Documents/Pesquisa/DATASET-256-LOW-DOSE/90_projections/"
target_dir = "/home/calves/Documents/Pesquisa/DATASET-256-LOW-DOSE/90_projs_target/"

if tipo == 1:
    model_src = "/home/calves/Documents/Pesquisa/models/GOOGLENET-model-UNETVGG"
    print("REDCNN")
elif tipo == 2:
    model_src = "/home/calves/Documents/Pesquisa/models/GOOGLENET-model-MSE"
    print("MSE")



means           = data_mean_value("training-final.csv", input_dir) / 255.

error_min = 0
error_max = 0.2
ex1 = "Tomo_000_slice_001.png"
ex2 = "Tomo_003_slice_025.png"
ssims = []
psnrs = []
rmses = []
ex1ssim = 0
ex1psnr = 0
ex1rmse = 0

ex2ssim = 0
ex2psnr = 0
ex2rmse = 0
def evaluate_img():

    test_data = dto.Tomographic_Dataset(csv_file="test-final.csv", phase='val', flip_rate=0, train_csv="training-final.csv",
                                    input_dir=input_dir, target_dir=target_dir)
    test_loader = DataLoader(test_data, batch_size=1)

    fcn_model = torch.load(model_src)
    n_tests = len(test_data.data)

    print("{} files for testing....".format(n_tests))

    if visao == 1:
        folder = "/home/calves/Documents/Pesquisa/PI/UNETGOOGLE/"
        if not os.path.exists(folder):
            os.makedirs(folder)
    elif visao == 2:
        folder = "/home/calves/Documents/Pesquisa/PI/MSEGOOGLE/"
        if not os.path.exists(folder):
            os.makedirs(folder)


    count = 0
    for iter, batch in enumerate(test_loader):
        name = batch['file'][0]
        print(name)
        dest = os.path.join(folder, name[0:len(name)-3])
        if not os.path.exists(dest):
            os.mkdir(dest)
        input = Variable(batch['X'].cuda())

        output = fcn_model(input)
        count = count + 1

        output = output.data.cpu().numpy()

        N, _, h, w = output.shape
        #y = output.transpose(0, 2, 3, 1).reshape(-1, n_class).argmax(axis=1).reshape(N, h, w)
        y = output[0, 0, :, :]
        target = batch['Y'].cpu().numpy().reshape(N, h, w)


        io.imsave(dest + '/residual.png', y)

        img_batch = batch['X']
        img_batch = img_batch + means

        grid = utils.make_grid(img_batch)
        x = grid.numpy()[::-1].transpose((1, 2, 0))

        io.imsave(dest + '/input.png', x)

        destplus = ""
        if visao == 1:
            destplus = "/home/calves/Documents/Pesquisa/PI/UNETexGOOGLE/"
        elif visao == 2:
            destplus = "/home/calves/Documents/Pesquisa/PI/MSEexGOOGLE/"

        if not os.path.exists(destplus):
            os.makedirs(destplus)
        io.imsave(destplus + name, y)
        if name == ex1:
            val_min = np.min(target[0, :, :])
            val_max = np.max(target[0, :, :])
        soma_ssim = ssim(y, target[0, :, :], data_range=val_max - val_min)
        soma_psnr = psnr(y, target[0, :, :], data_range=val_max - val_min)
        soma_rmse = rmse(y, target[0, :, :])
        ssims.append(soma_ssim)
        psnrs.append(soma_psnr)
        rmses.append(soma_rmse)
        y_temp = target[0, :, :] - y
        y_temp[y_temp > error_max] = error_max
        y_temp = (y_temp - error_min) / (error_max - error_min)
        dest = destplus
        cm = plt.get_cmap('hot')
        colored_image = cm(y_temp) * 255
        if name == ex1:
            ex1ssim = soma_ssim
            ex1psnr = soma_psnr
            ex1rmse = soma_rmse
            io.imsave(dest + name, colored_image)
        if name == ex2:
            ex2ssim = soma_ssim
            ex2psnr = soma_psnr
            ex2rmse = soma_rmse
            io.imsave(dest + name, colored_image)

        print("Média SSIM:", np.mean(ssims))
        print("STD SSIM:", np.std(ssims))

        print("Média PSNR:", np.mean(psnrs))
        print("STD PSNR:", np.std(psnrs))

        print("Média RMSE:", np.mean(rmses))
        print("STD RMSE:", np.std(rmses))


        count = count + 1

    print("SSIM do exemplo 1:", ex1ssim)
    print("PSNR do exemplo 1:", ex1psnr)
    print("RMSE do exemplo 1:", ex1rmse)

    print("SSIM do exemplo 2:", ex2ssim)
    print("PSNR do exemplo 2:", ex2psnr)
    print("RMSE do exemplo 2:", ex2rmse)




        #print("executed {} of {}\n".format(iter,len(test_loader)))

    #print("mean: {}".format(np.mean(execution_time[1:n_tests])))
    #print("std: {}".format(np.std(execution_time[1:n_tests])))


if __name__ == "__main__":
    evaluate_img()

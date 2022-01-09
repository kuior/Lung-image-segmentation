from model import *
from data import *
from keras.utils.training_utils import multi_gpu_model

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

#data_gen_args = dict()
data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')

myGene = trainGenerator(32,'data/membrane/train','image2','label2',data_gen_args,save_to_dir = None)
testGene = testGenerator("data/membrane/test2_new")

model = unet()
model = multi_gpu_model(model, 4)
model.compile(optimizer = Adam(lr = 1e-4), loss = weighted_binary_crossentropy_loss(5.28), metrics = [dice_coef])
#model.compile(optimizer = Adam(lr = 1e-5), loss = dice_coef_loss(), metrics = [dice_coef])
model.fit_generator(myGene,steps_per_epoch=5325,epochs=5,verbose=1)

results = model.predict_generator(testGene,56806,verbose=1)
saveResult("data/membrane/test2_pre_WBCE",results)

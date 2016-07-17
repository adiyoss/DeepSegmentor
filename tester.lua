----------------------------------------------------------------------
require('mobdebug').start()
require 'torch'   -- torch
require 'nn'
require 'rnn'
require 'xlua'
dofile ('model/structured_hinge_loss.lua') -- new loss function
d = dofile('common/data.lua')
m = dofile('model/model.lua')
tr = dofile('train.lua')

----------------------------------------------------------------------

x_path = '/Users/yossiadi/Projects/deep_audio_segmentation/code/segmentor/data/t/'
y_path = '/Users/yossiadi/Projects/deep_audio_segmentation/code/segmentor/data/t/'

d:new()
d_x, d_y, d_fn = d:read_data(x_path, y_path, 13)

for i=1, #d_fn do
  print(d_fn[i])
end
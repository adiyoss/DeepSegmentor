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
eval = dofile('evaluate.lua')


----------------------------------------------------------------------
if not opt then
   print '==> processing options'
   cmd = torch.CmdLine()
   cmd:text()
   cmd:text('Bi-RNN for Audio Segmentation')
   cmd:text()
   cmd:text('Options:')

   -- general
   cmd:option('-seed', 1234, 'the seed to generate numbers')
   -- data
   cmd:option('-features_path', 'data/synthetic/k=4/train/', 'the path to the features file')
   cmd:option('-labels_path', 'data/synthetic/k=4/train/', 'the path to the labels file')
   cmd:option('-t7_filename', 'train.t7', 'the .t7 filename')
   cmd:option('-input_dim', 9, 'the input size')
   cmd:option('-n_frames', 4, 'the number of frames to concatenate')
   -- model
   cmd:option('-model_path', 'results/model.net', 'the path to the model')
   -- train
   cmd:option('-type', 'double', 'data type: double | cuda')
   cmd:option('-x_suffix', '.data', 'the suffix of the data files')
   cmd:option('-y_suffix', '.labels', 'the suffix of the label files')
   
   cmd:text()
   opt = cmd:parse(arg or {})
end

----------------------------------------------------------------------
-- for CUDA
if opt.type == 'cuda' then
   print('==> switching to CUDA')
   require 'cunn'
   torch.setdefaulttensortype('torch.FloatTensor')
end

-- define parameters
local time = 0

-- ============================ load the data ============================ --
print '==> loading data set'
d:new(opt.x_suffix, opt.y_suffix)
x_test, y_test, f_n_test = d:read_data(opt.features_path, opt.labels_path, opt.input_dim, opt.t7_filename)

-- applying z-score normalization
mue, sigma = d:calc_z_score_params(x_test)
x_test = d:normalize(x_test, mue[1][1], sigma[1][1])

-- concat frames
x_test, y_test = d:concat_frames(x_test, y_test, opt.n_frames)

-- ========== define the model, loss and optimization technique ========== --
print '==> define loss'
criterion = nn.StructuredHingeLoss(opt.eps)
print(criterion)

print '==> build the model and initialize weights'
model = torch.load(opt.model_path)
print(model)

-- evaluate mode
model:evaluate()
print '==> first, evaluate on validation set'
loss, score, _ = eval:evaluate(model, criterion, x_test, y_test, f_n_test, true)

print('\n==> average score: ' .. score)
print('==> average cumulative loss: ' .. loss)
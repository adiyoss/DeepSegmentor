----------------------------------------------------------------------
--require('mobdebug').start()
require 'torch'   -- torch
require 'nn'
require 'rnn'
require 'xlua'
dofile ('model/structured_hinge_loss.lua') -- new loss function
d = dofile('common/data.lua')
m = dofile('model/model.lua')
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
   cmd:option('-features_path', 'data/vot/bb_pos/test/', 'the path to the features file')
   cmd:option('-labels_path', 'data/vot/bb_pos/test/', 'the path to the labels file')
   cmd:option('-test_file', 'test.t7', 'the t7 file that contains the data')
   cmd:option('-input_dim', 63, 'the input size')
   -- train
   cmd:option('-model', 'results/model.net', 'the path to the model directory')
   cmd:option('-type', 'double', 'data type: double | cuda')
   -- loss
   cmd:option('-eps', 5, 'the tolerance value for the loss function')
   
   cmd:text()
   opt = cmd:parse(arg or {})
end
----------------------------------------------------------------------

torch.manualSeed(opt.seed)
local eps = opt.eps

d:new()
print '==> Loading data set'
x, y, f_n = d:read_data(opt.features_path, opt.labels_path, opt.input_dim, opt.test_file)

print '==> define loss'
criterion = nn.StructuredHingeLoss(eps)
print(criterion)

print '==> loading model'
model = torch.load(opt.model)
print(model)

print '==> predict '
local loss, score, durations = eval:evaluate(model, criterion, x, y, f_n, true)
eval:plot_vot_stats(durations)
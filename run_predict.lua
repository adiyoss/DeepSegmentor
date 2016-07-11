----------------------------------------------------------------------
--require('mobdebug').start()
require 'torch'   -- torch
require 'nn'
require 'rnn'
require 'xlua'
dofile ('model/structured_hinge_loss.lua') -- new loss function
d = dofile('common/data.lua')
m = dofile('model/model.lua')

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
   cmd:option('-features_path', 'data/features/word_duration/x.t7', 'the path to the features file')
   cmd:option('-labels_path', 'data/labels/word_duration/y.t7', 'the path to the labels file')
   cmd:option('-input_dim', 13, 'the input size')
   -- train
   cmd:option('-model', 'results/model.net', 'the path to the model directory')
   cmd:option('-type', 'double', 'data type: double | cuda')
   -- loss
   cmd:option('-eps', 2, 'the tolerance value for the loss function')
   
   cmd:text()
   opt = cmd:parse(arg or {})
end
----------------------------------------------------------------------

torch.manualSeed(opt.seed)
local eps = 2

d:new()
print '==> Loading data set'
x, y = d:read_data_word_duration(opt.features_path, opt.labels_path, opt.input_dim)

print '==> define loss'
criterion = nn.StructuredHingeLoss(eps)
print(criterion)

print '==> loading model'
model = torch.load(opt.model)
print(model)

print '==> predict '
local cumulative_loss = 0
model:evaluate()
for t =1,#x do
    local output = model:forward(x[t])
    local score, onset, offset = criterion:predict(output)
    local loss = criterion:task_loss(y[t], {onset, offset})
    cumulative_loss = cumulative_loss + loss
    print('Score: ' .. score .. ', y_hat: [' .. onset .. ', ' .. offset ..'], y: [' .. y[t][1] .. ', ' .. y[t][2] .. '], loss: ' .. loss)    
end
print('\nAverage cumulative loss: ' .. cumulative_loss / #x)


----------------------------------------------------------------------
--require('mobdebug').start()
require 'torch'   -- torch
require 'nn'
require 'rnn'
require 'xlua'
dofile ('model/structured_hinge_loss.lua') -- new loss function
d = dofile('common/data.lua')
m = dofile('model/model.lua')
tr = dofile('train.lua')

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
   -- loss
   cmd:option('-eps', 1, 'the tolerance value for the loss function')
   -- model
   cmd:option('-hidden_size', 80, 'the hidden size')
   cmd:option('-dropout', 0.5, 'dropout rate')
   -- train
   cmd:option('-save', 'results', 'subdirectory to save/log experiments in')
   cmd:option('-plot', false, 'live plot')
   cmd:option('-optimization', 'ADAGRAD', 'optimization method: SGD | ADAM | ADAGRAD | RMSPROP | ADADELTA')
   cmd:option('-clipping', 5, 'gradient clipping in the range of [-5, 5]')
   cmd:option('-learningRate', 0.01, 'learning rate at t=0')
   cmd:option('-weightDecay', 0, 'weight decay (SGD only)')
   cmd:option('-momentum', 0.9, 'momentum (SGD only)')
   cmd:option('-type', 'double', 'data type: double | cuda')
   cmd:option('-n_epochs', 30, 'the number of epochs')
   
   cmd:text()
   opt = cmd:parse(arg or {})
end
----------------------------------------------------------------------
torch.manualSeed(opt.seed)

d:new()
print '==> Loading data set'
x, y = d:read_data_word_duration(opt.features_path, opt.labels_path, opt.input_dim)

print '==> define loss'
criterion = nn.StructuredHingeLoss(opt.eps)
print(criterion)

print '==> build the model and initialize weights'
method = 'xavier'
model = m:build_model(opt.input_dim, opt.hidden_size, opt.dropout, method)
print(model)

print '==> configuring optimizer'
tr:new(opt.type, opt.clipping)
tr:set_optimizer(opt.optimization)

-- Retrieve parameters and gradients
if model then
  parameters, gradParameters = model:getParameters()
end

print '==> training! '
local time = 0
local iteration = 1  -- epoch tracker

while iteration <= opt.n_epochs do
  model:training()
  print("==> online epoch # " .. iteration)
  for t =1,#x do    
    xlua.progress(t, #x)
    time = time + tr:train(x[t], y[t])
  end
  print("\n==> time to learn 1 sample = " .. (time*1000) .. 'ms')
  iteration = iteration + 1
  
  model:evaluate()
  local total_score = 0
  local cumulative_loss = 0
  for t =1,#x do
      local output = model:forward(x[t])
      local score, onset, offset = criterion:predict(output)
      total_score = total_score + score
      local loss = criterion:task_loss(y[t], {onset, offset})
      cumulative_loss = cumulative_loss + loss
      --print('Score: ' .. score .. ', y: [' .. y[t][1] .. ', ' .. y[t][2] .. '], y hat: [' .. onset .. ', ' .. offset .. ']')
  end
  print('Total Score: ' .. total_score / #x)  
  print('Cumulative Loss: ' .. cumulative_loss / #x .. '\n')  
end
torch.save('results/model.net', model)

--[[
model:evaluate()
for t =1,#x do
    local output = model:forward(x[t])
    local score, onset, offset = criterion:predict(output)
    print('Score: ' .. score .. ', onset: ' .. onset .. ', offset: ' .. offset)
end
]]--


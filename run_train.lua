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
   cmd:option('-features_path', 'data/word_duration/t7/x.t7', 'the path to the features file')
   cmd:option('-labels_path', 'data/word_duration/t7/y.t7', 'the path to the labels file')
   cmd:option('-input_dim', 13, 'the input size')
   -- loss
   cmd:option('-eps', 10, 'the tolerance value for the loss function')
   -- model
   cmd:option('-hidden_size', 100, 'the hidden size')
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
   cmd:option('-patience', 5, 'the number of epochs to be patience')
   
   cmd:text()
   opt = cmd:parse(arg or {})
end
----------------------------------------------------------------------
-- create loggers
lossLogger = optim.Logger(paths.concat(opt.save, 'loss.log'))
scoreLogger = optim.Logger(paths.concat(opt.save, 'score.log'))

-- set seed and save the parameters for reproducibility
torch.manualSeed(opt.seed)
paramsLogger = io.open(paths.concat(opt.save, 'params.log'), 'w')
for key, value in pairs(opt) do
  paramsLogger:write(key .. ': ' .. tostring(value) .. '\n')
end
paramsLogger:close()

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
local iteration = 1  -- for early stopping
local epoch = 1  -- epoch tracker
local best_loss = 9999999
local best_score = -1
local loss = -1

while loss < best_loss or iteration <= opt.patience do
  -- training
  model:training()
  print("==> online epoch # " .. epoch)
  for t =1,#x do    
    xlua.progress(t, #x)
    time = time + tr:train(x[t], y[t])
  end
  print("\n==> time to learn 1 sample = " .. (time*1000) .. 'ms')
  epoch = epoch + 1
  
  -- evaluating
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
  loss = cumulative_loss / #x
  print('Total Score: ' .. total_score / #x)  
  print('Cumulative Loss: ' .. loss .. '\n')
  
  -- early stopping criteria
  if loss >= best_loss then 
    -- increase iteration number
    iteration = iteration + 1
    print('\n========================================')
    print('==> Loss did not improved, iteration: ' .. iteration)
    print('========================================\n')
  else
    -- update the best loss value
    best_loss = loss  
    best_score = total_score
    
    -- save/log current net
    local filename = paths.concat(opt.save, 'model.net')
    os.execute('mkdir -p ' .. sys.dirname(filename))
    print('==> saving model to '..filename)    
    torch.save(filename, model)
    iteration = 1
  end
  -- update logger/plot
  lossLogger:add{['% loss (train set)'] = loss}
  scoreLogger:add{['% score (train set)'] = total_score}
end

lossLogger:add{['% loss (train set)'] = best_loss}
scoreLogger:add{['% score (train set)'] = best_score}

--[[
model:evaluate()
for t =1,#x do
    local output = model:forward(x[t])
    local score, onset, offset = criterion:predict(output)
    print('Score: ' .. score .. ', onset: ' .. onset .. ', offset: ' .. offset)
end
]]--


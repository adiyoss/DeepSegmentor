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
   cmd:option('-features_path', 'data/vot/bb_pos/', 'the path to the features file')
   cmd:option('-labels_path', 'data/vot/bb_pos/', 'the path to the labels file')
   cmd:option('-input_dim', 51, 'the input size')
   -- loss
   cmd:option('-eps', 5, 'the tolerance value for the loss function')
   -- model
   cmd:option('-hidden_size', 100, 'the hidden size')
   cmd:option('-dropout', 0.1, 'dropout rate')
   cmd:option('-n_layers', 2, 'the number of layers')
   -- train
   cmd:option('-save', 'results/vot/bb_pos/model_14/', 'subdirectory to save/log experiments in')
   cmd:option('-plot', false, 'live plot')
   cmd:option('-optimization', 'SGD', 'optimization method: SGD | ADAM | ADAGRAD | RMSPROP | ADADELTA')
   cmd:option('-clipping', 5, 'gradient clipping in the range of [-n, n]')
   cmd:option('-learningRate', 0.1, 'learning rate at t=0')
   cmd:option('-weightDecay', 0, 'weight decay (SGD only)')
   cmd:option('-momentum', 0.9, 'momentum (SGD only)')
   cmd:option('-type', 'double', 'data type: double | cuda')
   cmd:option('-patience', 5, 'the number of epochs to be patience')
   cmd:option('-x_suffix', '.txt', 'the suffix of the data files')
   cmd:option('-y_suffix', '.labels', 'the suffix of the label files')
   
   cmd:text()
   opt = cmd:parse(arg or {})
end

----------------------------------------------------------------------
-- define parameters
local time = 0
local score = -1
local loss = -1
train_folder = 'train/'
test_folder = 'test/'

-- for CUDA
if opt.type == 'cuda' then
   print('==> switching to CUDA')
   require 'cunn'
   torch.setdefaulttensortype('torch.FloatTensor')
end

-- ========== create loggers and save the parameters ========== --
lossLogger = optim.Logger(paths.concat(opt.save, 'loss.log'))
scoreLogger = optim.Logger(paths.concat(opt.save, 'score.log'))

-- set seed and save the parameters for reproducibility
torch.manualSeed(opt.seed)
paramsLogger = io.open(paths.concat(opt.save, 'params.log'), 'w')
for key, value in pairs(opt) do
  paramsLogger:write(key .. ': ' .. tostring(value) .. '\n')
end
paramsLogger:close()


-- ============================ load the data ============================ --
d:new(opt.x_suffix, opt.y_suffix)
print '==> Loading data set'
x_train, y_train, f_n_train = d:read_data(paths.concat(opt.features_path, train_folder), paths.concat(opt.labels_path, train_folder), opt.input_dim, 'train.t7')
x_test, y_test, f_n_test = d:read_data(paths.concat(opt.features_path, test_folder), paths.concat(opt.labels_path, test_folder), opt.input_dim, 'test.t7')

-- ========== define the model, loss and optimization technique ========== --
print '==> define loss'
criterion = nn.StructuredHingeLoss(opt.eps)
print(criterion)

print '==> build the model and initialize weights'
method = 'xavier'
model = m:build_model(opt.input_dim, opt.hidden_size, opt.dropout, method, opt.n_layers)
print(model)

print '==> configuring optimizer'
tr:new(opt.type, opt.clipping)
tr:set_optimizer(opt.optimization)

-- Retrieve parameters and gradients
if model then
  parameters, gradParameters = model:getParameters()
end

-- ============================= training ================================ --
print '==> training! '
local epochs = 20

-- loop for the number of epochs
for e =1, epochs do
  -- training mode
  model:training()
  
  -- do full epoch
  print("==> online epoch # " .. e)
  for t=1, #x_train do
    xlua.progress(t, #x_train)
    time = time + tr:train(x_train[t], y_train[t])
  end  
  print("\n==> time to learn 1 sample = " .. (time*1000) .. 'ms')
  
  -- evaluate mode
  model:evaluate()
  print '==> evaluating on validation set'
  loss, score = eval:evaluate(model, criterion, x_train, y_train, f_n_train)
  
  print('\n==> Average score: ' .. score)
  print('==> Average cumulative loss: ' .. loss)
  
  -- save/log current net
  local filename = paths.concat(opt.save, 'model.net')
  os.execute('mkdir -p ' .. sys.dirname(filename))
  print('==> loss improved, saving model to '..filename)
  torch.save(filename, model)
  
  -- update loggers
  lossLogger:add{['% loss (train set)'] = loss}
  scoreLogger:add{['% score (train set)'] = score}
end

-- ============================== testing ================================ --
print '==> evaluating on test set'
local loss_test, score_test, durations_test = eval:evaluate(model, criterion, x_test, y_test, f_n_test, true)
eval:plot_vot_stats(durations_test)
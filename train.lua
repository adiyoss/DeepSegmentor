-------------------------------
require 'torch'
require 'xlua'
require 'optim'
-------------------------------
local trainer = {}

-- C'Tor
function trainer:new(data_type, clipping)  
  -- CUDA?
  if data_type == 'cuda' then
     model:cuda()
     criterion:cuda()
     
     -- Log results to files
     trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))
     testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))
  end
  self.clipping = clipping
  -- Log results to files
  gradLogger = optim.Logger(paths.concat(opt.save, 'grad.log'))
--  testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))
end

function trainer:set_optimizer(optimization)  
  if optimization == 'SGD' then
     self.optim_state = {
        learningRate = opt.learningRate,
        weightDecay = opt.weightDecay,
        momentum = opt.momentum,
        learningRateDecay = 1e-7
     }
     self.optim_method = optim.sgd

  elseif opt.optimization == 'ADAM' then
     self.optim_state = {
        learningRate = opt.learningRate
     }
     self.optim_method = optim.adam
     
  elseif opt.optimization == 'ADAGRAD' then
     self.optim_state = {
        learningRate = opt.learningRate,
     }
     self.optim_method = optim.adagrad
     
  elseif opt.optimization == 'ADADELTA' then
    self.optim_state = {
        learningRate = opt.learningRate,
     }
     self.optim_method = optim.adadelta
     
  elseif opt.optimization == 'RMSPROP' then
    self.optim_state = {
        learningRate = opt.learningRate,
     }
     self.optim_method = optim.rmsprop
  else
     error('unknown optimization method')
  end
end

function trainer:train(train_x, train_y)
  
  -- local vars
  local time = sys.clock()

  -- create closure to evaluate f(X) and df/dX
  local feval = function(x)
    
                   -- reset gradients
                   gradParameters:zero()

                   -- f is the average of all criterions
                   local f = 0

                   -- estimate f
                   local output = model:forward(train_x)
                   local err = criterion:forward(output, train_y)
                   f = f + err

                   -- estimate df/dW
                   local df_do = criterion:backward(output, train_y)
                   model:backward(train_x, df_do)
                   
                   -- gradient clipping
                   gradParameters:clamp(-self.clipping, self.clipping)                               
                   
                   -- normalize gradients and f(X)
                   gradParameters:div(#train_x)
                   f = f/#train_x
                   
                   -- update logger/plot
                   -- tracking the gradients
                   gradLogger:add{['% grad norm (train set)'] = torch.norm(gradParameters)}
                   
                   -- return f and df/dX
                   return f, gradParameters
                end
      
  -- optimize on current mini-batch
  self.optim_method(feval, parameters, self.optim_state)

 -- time taken
 time = sys.clock() - time
 time = time / #train_x
 
 return time
end
  
return trainer
-------------------------------

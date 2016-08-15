-------------------------------
require 'torch'   -- torch
require 'nn'      -- provides all sorts of loss functions
dofile ('structured_hinge_loss.lua') -- new loss function
-------------------------------

local loss = {}

-- C'Tor
function loss:new(eps)  
  self.eps = eps
end

print '==> define loss'
if opt.loss == 'nll' then  
   criterion = nn.StructuredHingeLoss(5)
   --criterion = nn.SequencerCriterion(nn.ClassNLLCriterion(weights))
else
   error('unknown -loss')
end

----------------------------------------------------------------------
print '==> here is the loss function:'
print(criterion)

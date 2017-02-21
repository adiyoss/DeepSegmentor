local StructLayer, Parent = torch.class('nn.StructLayer', 'nn.Module')

require('common/weight_init.lua')

-- C'TOR
function StructLayer:__init(dim)
  Parent.__init(self)
  self.weight = torch.zeros(1, dim)
  self.gradWeight = torch.randn(1, dim)
  self.dim = dim
end

-- compute the forward pass
function StructLayer:updateOutput(input)
  -- support mini-batches
  if input:size(2) == 1 then
    self.output:resize(input:size(1)):zero()  
  else    
    self.output:resize(input:size(1), input:size(2)):zero()
  end
  
  -- W * x for all timesteps
  for i=1,input:size(1) do
    self.output[i] = self.weight * input[i]:t()    
  end
  return self.output
end

-- compute the gradient w.r.t the input
-- df / dx
function StructLayer:updateGradInput(input, gradOutput)
  self.gradInput = torch.zeros(input:size())
  local size = #gradOutput / 2
  for i=1, size do
    self.gradInput[tonumber(gradOutput[i][1])]:add(self.weight) --> w for phi(x, y_hat)
    self.gradInput[tonumber(gradOutput[i+size][1])]:add(torch.mul(self.weight, -1)) --> -w for phi(x, y)
  end
  return self.gradInput  
end

-- compute the gradient w.r.t the parameters
-- df / dw
function StructLayer:accGradParameters(input, gradOutput)  
  local phi_y_hat = torch.zeros(1, self.dim)
  local phi_y = torch.zeros(1, self.dim)
  local size = #gradOutput / 2
  for i=1, size do
    phi_y_hat:add(input[tonumber(gradOutput[i][1])])
    phi_y:add(input[tonumber(gradOutput[i+size][1])])
  end  
  self.gradWeight:add(phi_y_hat:csub(phi_y)) --> phi_y_hat - phi_y
end

function StructLayer:parameters()
  return {self.weight}, {self.gradWeight}
end
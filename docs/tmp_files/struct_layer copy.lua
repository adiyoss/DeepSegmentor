local StructLayer, Parent = torch.class('nn.StructLayer', 'nn.Module')

function StructLayer:__init(dim)
  Parent.__init(self)
  self.weight = torch.zeros(1, dim)
  self.gradWeight = torch.randn(1, dim)
  self.dim = dim
  
  Parent:reset()
end

function StructLayer:updateOutput(input)     
  self.output:resize(#input)
  self.output:zero()    
  for i=1,#input do
    self.output[i] = self.weight * input[i]:t()    
  end
  return self.output
end

function StructLayer:updateGradInput(input, gradOutput)
  self.gradInput = {}
  for k, v in pairs(input) do self.gradInput[k] = torch.zeros(1, self.dim) end
  self.gradInput[tonumber(gradOutput[1][1])]:add(self.weight)
  self.gradInput[tonumber(gradOutput[2][1])]:add(self.weight)
  self.gradInput[tonumber(gradOutput[3][1])]:add(torch.mul(self.weight, -1))
  self.gradInput[tonumber(gradOutput[4][1])]:add(torch.mul(self.weight, -1))
  return self.gradInput  
end

function StructLayer:accGradParameters(input, gradOutput)  
  local phi_y_hat = torch.zeros(1, self.dim)
  local phi_y = torch.zeros(1, self.dim)
  phi_y_hat:add(input[tonumber(gradOutput[1][1])], input[tonumber(gradOutput[2][1])])
  phi_y:add(input[tonumber(gradOutput[3][1])], input[tonumber(gradOutput[4][1])])  
  self.gradWeight:add(phi_y_hat:csub(phi_y)) --self.gradWeight = phi_y_hat - phi_y  
end

function StructLayer:parameters()
  return {self.weight}, {self.gradWeight}
end
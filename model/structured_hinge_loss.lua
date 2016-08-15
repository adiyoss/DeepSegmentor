local StructuredHingeLoss, Parent = torch.class('nn.StructuredHingeLoss', 'nn.Criterion')

-- C'TOR
function StructuredHingeLoss:__init(epsilon)
   Parent.__init(self)
   self.eps = epsilon or 0
   self.onset = 0
   self.offset = 0
   self.gradInput = {torch.Tensor(1), torch.Tensor(1), torch.Tensor(1), torch.Tensor(1)}
   -- consts
   self.MIN_GAP = 1
   self.MIN_SIZE = 4
   
end

-- compute forward pass
-- loss augmented inference
function StructuredHingeLoss:updateOutput(input, y)  
  self.onset = 1
  self.offset = 2
  self.output = input[self.onset] + input[self.offset] + self:task_loss({self.onset, self.offset}, y, self.eps)
  
  -- TODO add more constraints
  for i=self.MIN_GAP,input:size(1) do
    for j=i+self.MIN_SIZE,input:size(1) do
      local tmp = input[i] + input[j] + self:task_loss({i, j}, y, self.eps)
      if tmp > self.output then
        self.output = tmp
        self.onset = i
        self.offset = j
      end
    end
  end
  return self.output
end

-- compute the gradientes w.r.t the input
-- df / dx
function StructuredHingeLoss:updateGradInput(input, y)
  self.gradInput[1][1] = self.onset
  self.gradInput[2][1] = self.offset
  self.gradInput[3][1] = y[1]
  self.gradInput[4][1] = y[2]
  return self.gradInput
end

-- task loss / cost function
function StructuredHingeLoss:task_loss(y, y_hat, eps)
  local loss = 0
  local e = (eps or self.eps)
  
  for i=1,#y do
    loss = loss + math.max(0, torch.abs(y[i] - y_hat[i]) - e)
  end
  return loss / #y
end

-- inference
function StructuredHingeLoss:predict(input)  
  local onset = 1
  local offset = 2
  local output = input[onset] + input[offset]

  -- TODO add more constraints
  for i=self.MIN_GAP,input:size(1) do
    for j=i+self.MIN_SIZE,input:size(1) do
      local tmp = input[i] + input[j]
      if tmp > output then
        output = tmp
        onset = i
        offset = j
      end
    end
  end
  return output, onset, offset  
end
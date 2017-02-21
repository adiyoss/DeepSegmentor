local StructuredHingeLoss, Parent = torch.class('nn.StructuredHingeLoss', 'nn.Criterion')

-- C'TOR
function StructuredHingeLoss:__init(epsilon)
   Parent.__init(self)
   self.eps = epsilon or 0
   self.onsets = {}
   
   -- consts
   self.MIN_GAP = 1
   self.MIN_SIZE = 4
end

-- compute forward pass
-- loss augmented inference
function StructuredHingeLoss:updateOutput(input, y)      
  self.onsets = {}
  table.insert(self.onsets, 1)
  table.insert(self.onsets, 2)
  
  self.output = input[self.onsets[1]] + input[self.onsets[2]] + self:task_loss({self.onsets[1], self.onsets[2]}, y, self.eps)
  
  for i=self.MIN_GAP,input:size(1) do
    for j=i+self.MIN_SIZE,input:size(1) do
      local tmp = input[i] + input[j] + self:task_loss({i, j}, y, self.eps)
      if tmp > self.output then
        self.output = tmp
        self.onsets[1] = i
        self.onsets[2] = j
      end
    end
  end
  return self.output
end

-- compute the gradientes w.r.t the input
-- df / dx
function StructuredHingeLoss:updateGradInput(input, y)
  self.gradInput = {}  
  for i=1, 2 * #y do    
    table.insert(self.gradInput, torch.Tensor(1))
  end
  for i=1, #self.onsets do
    self.gradInput[i][1] = self.onsets[i]
    self.gradInput[i + #self.onsets][1] = y[i]
  end
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
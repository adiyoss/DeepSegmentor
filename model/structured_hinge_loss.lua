local StructuredHingeLoss, Parent = torch.class('nn.StructuredHingeLoss', 'nn.Criterion')

-- C'TOR
function StructuredHingeLoss:__init(epsilon)
   Parent.__init(self)
   self.eps = epsilon or 0
   self.onsets = {}
   
   -- consts
   self.MIN_GAP = 5
   self.MIN_SIZE = 4
end

-- compute forward pass
-- perform loss augmented inference
function StructuredHingeLoss:updateOutput(input, y)      
  self.output = 0
  self.onsets = {}  
  
  -- loop over all the targets' components
  for i=1, #y do
    -- init the best score and best onset
    local best_score = input[1] + (self:task_loss({y[1]}, {1}, self.eps) / 100)
    local best_onset = 1
    local start_idx = self.onsets[i-1] or 1
    -- if we reach the end of the sequence, set all other labels to the same label
    if start_idx + self.MIN_GAP < input:size(1) then
      -- loop over all possiable onsets to find the max value
      for j=start_idx + self.MIN_GAP, input:size(1) do
        local tmp = input[j] + (self:task_loss({y[i]}, {j}, self.eps) / 100)
        if best_score < tmp then
          best_score = tmp
          best_onset = j
        end
      end          
      self.output = self.output + best_score
      table.insert(self.onsets, best_onset)
    else
      local l_score = input[input:size(1)] + (self:task_loss({y[i]}, {input:size(1)}, self.eps) / 100)
      local l_onset = input:size(1)
      
      self.output = self.output + l_score      
      table.insert(self.onsets, l_onset)
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
  return (loss / #y)
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
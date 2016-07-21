local utils = {}

-- local functions
local function isempty(s)
  return s == nil or s == '' or s == '\n'
end

-- calaulates the number of line in the file
function utils:get_num_of_lines(path)
  local i = 0
  local file = io.open(path, "r")
  if file then
    for line in file:lines() do
      if isempty(line) ~= true then
        i = i + 1
      end
    end
  else
    print("\n==>ERROR: can not open file.")    
  end  
  if file then
    file:close()
  end
  return i
end

-- load features from txt file
function utils:load_data(path, input_dim)
  local file = io.open(path, "r")
  num_of_examples = self:get_num_of_lines(path)
  
  local data = torch.Tensor(num_of_examples, 1, input_dim)
  --local data = {}
  if file then
    local i = 1
    for line in file:lines() do
      local j = 1
      --local tmp = torch.Tensor(1, input_dim)
      for str in string.gmatch(line, "(%S+)") do
        -- read only until input dim is reached
        if j > input_dim then
          break
        end
        data[i][1][j] = str
        j = j + 1
      end      
      -- support gpu processing
      if (j - 1) == input_dim then        
        if not opt or not opt.type then
          data[i] = data[i]:double()
        elseif opt.type == 'double' then
          data[i] = data[i]:double()
        elseif opt.type == 'cuda' then
          data[i] = data[i]:cuda()
        end
        i = i + 1
      end
    end
  else
    print("\n==>ERROR: can not open file.")    
  end 
  if file then
    file:close()
  end  
  return data
end

-- load labels from txt file
function utils:load_labels(path)
  local file = io.open(path, "r")
  local labels = {}
  if file then
    local is_first = 0
    for line in file:lines() do
      if is_first == 1 then
        for str in string.gmatch(line, "(%S+)") do
          table.insert(labels, str)
        end
      end
      is_first = is_first + 1
    end
  else
    print("\n==>ERROR: can not open file.")    
  end  
  if file then
    file:close()
  end
  return labels
end

-- converts a vector of indicators to a table of onsets
function utils:convert2pairs(y)
  onsets = {}
  for i=2,y:size(1) do
    if y[i] ~= y[i - 1] then
      table.insert(onsets, i)
    end
  end
  return onsets
end

-- argmax
function utils:argmax_1D(v)
   local length = v:size(1)
   assert(length > 0)

   -- examine on average half the entries
   local maxValue = torch.max(v)
   for i = 1, v:size(1) do
      if v[i] == maxValue then
         return i
      end
   end
end

return utils
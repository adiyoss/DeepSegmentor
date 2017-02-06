-------------------------------
require 'torch'
require 'nn'
require 'xlua'
utils = dofile('common/utils.lua')
logger = dofile('common/logger.lua')
-------------------------------

-------------------------------
local data = {}

-- C'Tor
function data:new(x_suffix, y_suffix)
  -- default value is .data and .labels
  if x_suffix ~= nil then
    self.x_suffix = x_suffix
  else
    self.x_suffix = '.txt'
  end
  if y_suffix ~= nil then
    self.y_suffix = y_suffix
  else
    self.y_suffix = '.labels'
  end
end

-- validate that for every .txt file there exists .textgrid file
function data:validate_data(path_x, path_y)
  local n_files = 0
  for file in paths.files(path_x) do
    if string.sub(file,-string.len(self.x_suffix))==self.x_suffix then
      local x_filename = paths.concat(path_x, file) 
      local y_filename = paths.concat(path_y, (string.sub(file, 0, string.len(file)-string.len(self.x_suffix)) .. self.y_suffix))
      if paths.filep(x_filename) and paths.filep(y_filename) then
        n_files = n_files + 1
      else
        logger:error('probelm with data, missing files: ' .. x_filename .. ' or ' .. y_filename .. '\n') 
      end
    end
  end
  return n_files
end

-- read data
function data:read_data(path_x, path_y, input_dim, t7_file)
  -- validation
  if not self.x_suffix or not self.y_suffix then
    logger:error('object was not initialized. call new() function before using data module.\n') 
  end
  
  local indicator = 1
  local x = {}
  local y = {}
  local f_n = {}
  
  -- check if the t7 exists
  -- if so read the data from this file is faster
  d_path = paths.concat(path_x, t7_file)
  if t7_file and paths.filep(d_path) then    
    all = torch.load(d_path)
    x = all[1]
    y = all[2]
    f_n = all[3]
  else
    local n_files = self:validate_data(path_x, path_y)
    for file in paths.files(path_x) do
      if string.sub(file,-string.len(self.x_suffix))==self.x_suffix then
        xlua.progress(indicator, n_files)
        indicator = indicator + 1
        local x_filename = paths.concat(path_x, file) 
        local y_filename = paths.concat(path_y, (string.sub(file, 0, string.len(file)-string.len(self.x_suffix)) .. self.y_suffix))
        local x_t = utils:load_data(x_filename, input_dim)
        local y_t = utils:load_labels(y_filename)
        table.insert(x, x_t)
        table.insert(y, y_t)      
        table.insert(f_n, file)
      end
    end 
  end
  
  -- save to t7 if the file is not exists
  if t7_file then
    if not paths.filep(d_path) then
      all = {}
      table.insert(all, x)
      table.insert(all, y)
      table.insert(all, f_n)
      print('saving the data to ' .. d_path .. ' for faster loading')
      torch.save(d_path, all)
    end
  end
  
  return x, y, f_n
end

function data:create_mini_batches(x, y, batch_size)
  local max = 0
  local input_dim = x[1]:size(3)
  -- find max
  for i=1 ,batch_size do
    max = (max < x[i]:size(1)) and x[i]:size(1) or max
  end
  
  local b_x, b_y = {}, {}
  local x_all = torch.zeros(max, batch_size, input_dim)  
  local y_all = {}
  for i=1 ,batch_size do
    x_all[{{}, {i}}][{{1, x[i]:size(1)}}] = x[i]
    y_all[i] = y[i]
  end
  table.insert(b_x, x_all)
  table.insert(b_y, y_all)
  return b_x, b_y
end

return data
-------------------------------

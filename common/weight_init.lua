--
-- Different weight initialization methods
--
-- > model = require('weight-init')(model, 'heuristic')
--
require("nn")


-- "Efficient backprop"
-- Yann Lecun, 1998
function w_init_heuristic(fan_in, fan_out)
   return math.sqrt(1/(3*fan_in))
end


-- "Understanding the difficulty of training deep feedforward neural networks"
-- Xavier Glorot, 2010
function w_init_xavier(fan_in, fan_out)
   return math.sqrt(2/(fan_in + fan_out))
end


-- "Understanding the difficulty of training deep feedforward neural networks"
-- Xavier Glorot, 2010
function w_init_xavier_caffe(fan_in, fan_out)
   return math.sqrt(1/fan_in)
end


-- "Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification"
-- Kaiming He, 2015
function w_init_kaiming(fan_in, fan_out)
   return math.sqrt(4/(fan_in + fan_out))
end

--[[
local function w_init(net, arg)
   -- choose initialization method
   local method = nil
   if     arg == 'heuristic'    then method = w_init_heuristic
   elseif arg == 'xavier'       then method = w_init_xavier
   elseif arg == 'xavier_caffe' then method = w_init_xavier_caffe
   elseif arg == 'kaiming'      then method = w_init_kaiming
   else
      assert(false)
   end

   for i = 1, #net.modules do
      local m = net.modules[i]
      dims = m.weight:size()
      m:reset(method(dims[1], dims[2]))
      if m.bias then
         m.bias:zero()
      end
   end
   return net
end
]]--

--return w_init
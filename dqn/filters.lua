require 'initenv'
require 'cutorch'
require 'image'
require 'xlua'
require 'NeuralQLearner'
require 'NeuralQLearnerISA'
require 'NeuralQLearnerDouble'

op = xlua.OptionParser('filters.lua -m <model.t7>')
op:option{'-m', '--model', action='store', dest='model', help='first layer visualization', default=''}
opt = op:parse()
op:summarize()

local network = torch.load(opt.model).best_model

local weight = network:get(2).weight:float()

local num_filters = 16
local sp_size = 8
local tp_size = 4

print('Min value '..weight:min())
print('Max value '..weight:max())
print('Mean value '..weight:mean())

weight = weight - weight:min()

local map = torch.Tensor(num_filters*sp_size+num_filters+1, tp_size*sp_size+tp_size+1):fill(1)

local row1 = 2
local row2 = row1+sp_size-1
for i = 1,num_filters do
	local col1 = 2
	local col2 = col1+sp_size-1
	for j = 1,tp_size do
		map[{{row1,row2},{col1,col2}}] = weight[{i,{j,{{}}}}]:resize(sp_size,sp_size)
		col1 = col1 + sp_size + 1
		col2 = col2 + sp_size + 1
	end
	row1 = row1 + sp_size + 1
	row2 = row2 + sp_size + 1
end

image.save('map.png', map)

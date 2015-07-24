--[[
Copyright (c) 2014 Google Inc.

See LICENSE file for full terms of limited license.
]]

require 'image'

if not dqn then
    require "initenv"
end

local cmd = torch.CmdLine()
cmd:text()
cmd:text('Evaluate Agent in Environment:')
cmd:text()
cmd:text('Options:')

cmd:option('-framework', '', 'name of training framework')
cmd:option('-env', '', 'name of environment to use')
cmd:option('-game_path', '', 'path to environment file (ROM)')
cmd:option('-env_params', '', 'string of environment parameters')
cmd:option('-pool_frms', '',
           'string of frame pooling parameters (e.g.: size=2,type="max")')
cmd:option('-actrep', 1, 'how many times to repeat action')
cmd:option('-random_starts', 0, 'play action 0 between 1 and random_starts ' ..
           'number of times at the start of each training episode')

cmd:option('-name', '', 'filename used for saving network and training history')
cmd:option('-network', '', 'reload pretrained network')
cmd:option('-agent', '', 'name of agent file to use')
cmd:option('-agent_params', '', 'string of agent parameters')
cmd:option('-seed', 1, 'fixed input seed for repeatable experiments')
cmd:option('-saveNetworkParams', false,
           'saves the agent network in a separate file')
cmd:option('-prog_freq', 5*10^3, 'frequency of progress output')
cmd:option('-save_freq', 5*10^4, 'the model is saved every save_freq steps')
cmd:option('-eval_freq', 10^4, 'frequency of greedy evaluation')
cmd:option('-save_versions', 0, '')

cmd:option('-steps', 10^5, 'number of training steps to perform')
cmd:option('-eval_steps', 10^5, 'number of evaluation steps')

cmd:option('-verbose', 2,
           'the higher the level, the more information is printed to screen')
cmd:option('-threads', 1, 'number of BLAS threads')
cmd:option('-gpu', -1, 'gpu flag')
cmd:option('-gpu_id',0,'GPU_ID')
cmd:option('-best', 1, 'best network')

cmd:option('-ale_host', '', 'ALE server host') 
cmd:option('-ale_port', 1567, 'ALE server port')
cmd:option('-ale_pass', '', 'password for ALE server') 
cmd:option('-ale_login', '', 'login for ALE server') 

cmd:text()

local opt = cmd:parse(arg)

if opt.best == 1 then
    opt.best = true
else
    opt.best = false
end

--- General setup.
local game_env, game_actions, agent, opt = setup(opt)

-- override print to always flush the output
local old_print = print
local print = function(...)
    old_print(...)
    io.flush()
end

local learn_start = agent.learn_start
local start_time = sys.clock()
local reward_counts = {}
local episode_counts = {}
local time_history = {}
local v_history = {}
local qmax_history = {}
local td_history = {}
local reward_history = {}
time_history[1] = 0

local total_reward
local nrewards
local nepisodes
local episode_reward
local estep
local total_steps

local screen, reward, terminal = game_env:getState()

--screen, reward, terminal = game_env:newGame()

total_reward = reward
nrewards = 0
nepisodes = 0
episode_reward = 0
estep = 0
total_steps = 0

function scale_forward(x, e, s)

    local x = x
    if x:dim() > 3 then
        x = x[1]
    end

    --x = image.rgb2y(x)
    --x = image.scale(x, 84, 84, 'bilinear')
    
    local str = string.format("../images/%d_%d.png", e, s) 
    image.save(str, x)
end

print('Begin evaluation')

local eval_time = sys.clock()

while 1 do

    estep = estep + 1

    if opt.verbose == 1 and reward ~= 0 then
        scale_forward(screen:float(), nepisodes + 1, estep)
        print('Step '..estep..' reward '..reward)
    end
 
    local action_index = agent:perceive(reward, screen, terminal, true, 0.05)

    -- Play game in test mode (episodes don't end when losing a life)
    screen, reward, terminal = game_env:step(game_actions[action_index])

    -- record every reward
    episode_reward = episode_reward + reward
    if reward ~= 0 then
       nrewards = nrewards + 1
    end

    total_steps = total_steps + 1
    
    if total_steps%1000 == 0 then collectgarbage() end

    if terminal == false and estep == 18000 then
        terminal = true
        print('Maximum steps without terminal!')
    end

    if terminal then
        total_reward = total_reward + episode_reward
        nepisodes = nepisodes + 1
        print(string.format('Episode %d = %d, average %.3f, steps %d, time %ds',
            nepisodes, episode_reward, total_reward / nepisodes, estep, sys.clock() - eval_time))
        episode_reward = 0
        estep = 0
        if nepisodes < 10 then
            screen, reward, terminal = game_env:nextRandomGame()
        else
            break
        end
    end
end

if estep > 0 then
    print(string.format('Last episode %d, time %ds', episode_reward, sys.clock() - eval_time))
end

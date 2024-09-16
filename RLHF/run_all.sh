#!/bin/bash

# 调用第一个脚本
bash scripts/13b-v1.5-336/train_reward_object.sh
# 调用第二个脚本
bash scripts/13b-v1.5-336/train_reward_attribute.sh
# 调用第三个脚本
bash scripts/13b-v1.5-336/train_reward_relation.sh

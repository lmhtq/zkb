#!/bin/bash
cd scheduler
python crawler.py &
sleep 1
python executavor.py &
sleep 1
python booster.py &



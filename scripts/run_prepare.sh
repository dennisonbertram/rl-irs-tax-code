#!/bin/bash
cd /Users/dennisonbertram/Develop/rl-irs-tax-code
python3 scripts/bulk_generate_training_data.py --prepare > /tmp/bulk_prepare.log 2>&1
echo "EXIT: $?" >> /tmp/bulk_prepare.log

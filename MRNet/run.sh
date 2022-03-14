TWCC_CLI_CMD=/home/ubuntu/.local/bin/twccli

echo "1. Creating CCS"
$TWCC_CLI_CMD mk ccs -itype "PyTorch" -img "pytorch-20.11-py3:latest" -gpu 1 -wait -json > ccs_res.log

CCS_ID=$(cat ccs_res.log | jq '.id')
echo "2. CCS ID:" $CCS_ID

echo "3. Checking GPU"
ssh -o "StrictHostKeyChecking=no" `$TWCC_CLI_CMD ls ccs -gssh -s $CCS_ID` "source /etc/profile.d/init.sh; nvidia-smi"

echo "4. RUN GPU"
ssh -o "StrictHostKeyChecking=no" `$TWCC_CLI_CMD ls ccs -gssh -s $CCS_ID` "source /etc/profile && python /home/tedwu430/tfcc/main.py"

echo "5. GC GPU"
$TWCC_CLI_CMD rm ccs -f -s $CCS_ID

echo "6. Checking CCS"
$TWCC_CLI_CMD ls ccs
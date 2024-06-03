! mkdir /scratch/cliao25

directory="/scratch/cliao25/domain_net"
if [ ! -d "$directory" ]; then
    echo "copying domain_net to scratch."
    cp ~/cliao25/data/domain_net.zip /scratch/cliao25
    unzip -DD -q  /scratch/cliao25/domain_net.zip -d  /scratch/cliao25/
else
    echo "domain_net exists."
fi
rm -f /scratch/cliao25/domain_net.zip

find /scratch/cliao25 -type f -exec touch {} +

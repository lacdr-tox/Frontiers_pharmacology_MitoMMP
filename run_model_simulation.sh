## uncomment the following piece of command to download and install miniconda
# wget -qO "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh" "miniconda3.sh";
# chmod +x miniconda3.sh;
# bash miniconda3.sh;

## uncomment the following line to create the env
# conda env create -f ./MitoMMP.yml

## uncomment the following line to activate the env
# conda activate MitoMMP;
# Fig 3B
python model_simulation.py --compound "Antimycin A";
# Fig 3C
python model_simulation.py --compound "Rotenone";
# Fig 4AB
python model_simulation_3d.py --compound 'Deguelin' 'Azoxystrobin';
# Fig 5
python model_simulation.py --compound "FCCP";
# Fig 6
python model_simulation.py --compound "Oligomycin";

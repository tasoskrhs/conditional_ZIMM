dir=`pwd`
dir="$(basename $dir)"
if [ "$dir" != "conditional_ZIMM" ]; then
	echo -e "should run it from conditional_ZIMM dir"
elif [ "${BASH_SOURCE[0]}" != "${0}" ]; then
	conda create -n tf2_env_py39 python=3.9 -y
	eval $(conda shell.bash hook)
	conda activate tf2_env_py39
	echo "installing requirements"
	pip install --user --requirement requirements.txt
else
	echo -e "remember to source this script: . ./setup.sh"
fi

dir=`pwd`
dir="$(basename $dir)"
if [ "$dir" != "cond_ZIMM" ]; then
	echo -e "should run it from cond_ZIMM dir"
elif [ "${BASH_SOURCE[0]}" != "${0}" ]; then
	conda create -n tf2_env_py36 python=3.6 -y
	eval $(conda shell.bash hook)
	conda activate tf2_env_py36
	echo "installing requirements"
	pip install --user --requirement requirements.txt
else
	echo -e "remeber to source this script: . ./setup.sh"
fi

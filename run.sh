#!/usr/bin/env sh

if [ $# -eq 0 ]; then
	echo "Usage: $0 <action>"
	exit 1
fi

python src/main.py --action "$1"

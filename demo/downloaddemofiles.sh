pip install gdown

if [ -d "img" ]; then
    if [ -f "img/image_demo.tiff" ]; then
	echo "demo image file already present"
    else
	gdown 10ecYfSZddkcRV1VqxgfNmyYYyC1UYF5Z -O img/image_demo.tiff
	echo "copied demo image file"
    fi
else
    mkdir img
    gdown 10ecYfSZddkcRV1VqxgfNmyYYyC1UYF5Z -O img/image_demo.tiff
    echo "copied demo image file"
fi
if [ -d "mask" ]; then
    if [ -f "mask/mask_demo.tiff" ]; then
	echo "demo mask file already present"
    else
	gdown 1iZ00HWhBpM-f2oe-EGaTevq2aAqW7jK8 -O mask/mask_demo.tiff
	echo "copied demo mask file"
    fi
else
    mkdir mask
    gdown 1iZ00HWhBpM-f2oe-EGaTevq2aAqW7jK8 -O mask/mask_demo.tiff
    echo "copied demo mask file"
fi


cat depfiles.txt | while read -r file; do 
    cat subjects.txt | while read -r subject; do 
        mkdir -p mri_dataset mri_processed_data/${subject}/modeling/surfaces \
            mri_processed_data/${subject}/concentrations \
            mri_processed_data/${subject}/segmentations
        subjectfile=$(echo $file | sed s/{subject}/sub-01/g);
        echo rsync -v $SOURCEDIR/$subjectfile ./$subjectfile
        rsync -v $SOURCEDIR/$subjectfile ./$subjectfile
    done
done

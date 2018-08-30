from os import listdir, remove

safe_files = ['feature_importance.csv']

for path in ['output/', 'output/pre_sub/']:
    for file in listdir(path):
        if file!='.gitkeep' and file!='pre_sub' and file!='.DS_Store':
            if file in safe_files:
                print('remove {}? (type \'yes\' to remove it)'.format(file))
                if input()=='yes':
                    remove(path+file)
                else:
                    continue
            else:
                remove(path+file)

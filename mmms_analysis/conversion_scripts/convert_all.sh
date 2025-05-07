#!/bin/bash

# Loop through all directories in the current directory
for dir in */ ; do
    # Remove trailing slash to get folder name
    folder_name="${dir%/}"

    # Construct expected Python filename
    script_file="convert_${folder_name}.py"

    # Check if the file exists in the directory
    if [[ -f "$dir/$script_file" ]]; then
        echo "Running $script_file in $folder_name"
        (
            cd "$dir" && python3 "$script_file"
        ) &
    else
        echo "No script found at $dir/$script_file"
    fi
done

# Wait for all background jobs to finish
wait

echo "All scripts completed."

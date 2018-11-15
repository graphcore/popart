# Utility functions used by the other scripts

# Try and return the number of logical processor the machine has
get_processor_count() {
    # Try GNU nproc...
    if ! count=$(nproc 2>/dev/null); then
        # Try macOS sysctl...
        if ! count=$(sysctl -n hw.ncpu 2>/dev/null); then
            # Just use a fixed number
            count=1
        fi
    fi

    echo ${count}
}


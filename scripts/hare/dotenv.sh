# Sourced by the scripts in this directory from the repository root.

# Value of the variable named $1: from the environment, else from .env. Prints
# nothing when it is set in neither; callers check.
env_or_dotenv() {
    local name=$1
    if [ -n "${!name:-}" ]; then
        printf '%s\n' "${!name}"
    elif [ -f .env ]; then
        sed -n "/^$name=/{s/^$name=//p;q;}" .env | tr -d "\"'"
    fi
}

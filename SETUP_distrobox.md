## Distrobox Setup (EECS106B)

Distrobox is a permissions-safe wrapper around Docker/Podman that makes it easy
to run a container like a user shell, with host files mounted and a familiar
developer workflow. This is the setup used by this package.

This README covers **Distrobox-based setup** for EECS106B.

### Step 1: Prereqs (host)
- Fork this repo and git clone your fork (all development should happen in this repo):
  ```
  git clone git@github.com:<your-username>/EECS106B.git
  cd EECS106B
  git submodule update --init --recursive
  ```
- Make sure you can authenticate with GitHub (preferably set up SSH keys).
- Add these to your host `~/.bashrc` (or `~/.zshrc`) (replace with your paths):
  ```
  export EECS106B_DIR="/path/to/EECS106B"
  source "$EECS106B_DIR/helpers"
  ```
  Or run this code to append them to `~/.bashrc`:
  ```
  if [[ "$(pwd)" == */EECS106B ]]; then cat <<'EOF' >> ~/.bashrc
  export EECS106B_DIR="$(pwd)"
  source "$EECS106B_DIR/helpers"
  EOF
  else echo "ERROR: run this from your EECS106B repo directory." >&2; return 1; fi
  ```
- Make sure to re-source using `source ~/.bashrc`

### Step 2: Create the distrobox
Run the helper function:
```
distrobox_create
```
This can take a while on first run (>= 30 minutes). If prompted to download an image from `docker.io`, answer "yes".
This pulls the `omnidrones:eecs` image (built from the Dockerfile) and creates the container with GPU/Vulkan passthrough. It also:
- Mounts your repo into `/workspace/omni_drones`
- Mounts a persistent venv from `"$EECS106B_DIR/.venv"` to `/opt/venv`
- Sets `HOME=/workspace/omni_drones` so the repo’s `.bashrc` is used inside the container

### Step 3: Enter the distrobox
```
distrobox_enter
```
This will move you inside the distrobox container. 

The first time you enter, you will need to install the required packages. Run `eecs106b_install` to install the packages. *Note* Expect some red warnings related to pip not respecting dependencies. This is expected, however you should not see any other errors.

#### What happens on entry (from repo `.bashrc`)
When you enter, `distrobox_enter` runs `bash -i`, which reads the repo’s `.bashrc`
because `HOME` is set to `/workspace/omni_drones`. That script:
- Exports `EECS106B_DIR` and `ISAACSIM_PATH`
- Sources `"$ISAACSIM_PATH/setup_conda_env.sh"`
- Activates `/opt/venv` if it exists, or creates it if missing
- Checks whether the required packages are installed and prints a reminder if not

#### Technical background (for people who care)
- The venv is **bind-mounted** from the host directory at `"$EECS106B_DIR/.venv"`
  into `/opt/venv` inside the container, so any pip installs persist across sessions.
- Because `HOME` points at `/workspace/omni_drones`, the repo’s `.bashrc` becomes the
  interactive shell config used on container entry.

### Step 4: Stop and remove the container
```
distrobox_end
```
This stops and removes the distrobox container and cleans up any leftover container.

### Quick start (verify your setup)
Inside the container:
```
cd /workspace/omni_drones/scripts
python train.py algo=ppo headless=true wandb.mode=disabled
```
If you see PPO training logs (e.g., average reward metrics), your setup is working.

# rp-live-demo

Live demo rig: a **headless Raspberry Pi 5 + USB webcam** runs a `.onnx` vision
transformer on the camera feed, annotates each frame with the top-k predictions,
and serves it as an **MJPEG-over-HTTP** stream. Your **laptop** views it in a
browser tab you can embed straight into your slides.

The Pi and laptop are joined by a **direct Ethernet cable** with static IPs, so
the demo never touches the venue's unreliable Wi-Fi.

```
[ USB webcam ] -> [ Raspberry Pi 5: capture -> ONNX ViT -> annotate -> MJPEG ]
                                       |
                     direct Ethernet cable (Pi runs DHCP)
                                       |
              [ Laptop browser: http://raspberrypi.local:8000/ ]  -> embed in slides
```

## 1. The physical link (do this once)

Plug an **ordinary Ethernet cable** straight between the Pi and the laptop — both
support Auto-MDI-X, so no crossover cable or switch is needed. The robust part is
making the **Pi run the tiny network itself**, so the laptop (especially Windows)
needs zero configuration.

### Recommended: the Pi hands out addresses (DHCP) + a hostname (mDNS)

Windows defaults its Ethernet adapter to "obtain an address automatically" and
fights manual static IPs (it usually falls back to a useless `169.254.x.x`
link-local address). So instead of configuring the laptop, make the Pi a DHCP
server. NetworkManager's built-in **shared** mode does exactly that: it gives the
Pi `10.42.0.1` and runs DHCP on the wired port.

```bash
# On the Pi — find the wired interface (usually eth0)
ip link
sudo nmcli con add type ethernet ifname eth0 con-name demo-link ipv4.method shared
sudo nmcli con up demo-link
```

Add mDNS so the laptop can reach the Pi *by name*, whatever address it gets:

```bash
sudo apt install -y avahi-daemon
sudo systemctl enable --now avahi-daemon
```

Now on the laptop: leave the Ethernet adapter on its defaults, plug in the cable,
wait a few seconds. It pulls an address from the Pi automatically and resolves
`raspberrypi.local` (substitute your Pi's hostname) via mDNS — built into
Windows 10 1809+, 11, macOS, and Linux. Verify from the laptop:

```powershell
ping raspberrypi.local
```

Why this beats static IPs: nothing to configure on the laptop, no `169.254.x.x`
APIPA fallback to fight, no firewall inbound rule (the laptop is only an outbound
HTTP client), and you reach the Pi by name even if the address changes. Keep the
Pi's Wi-Fi up for SSH/internet — only the cable carries the demo stream.

### Fallback: manual static IPs

For a locked-down corporate laptop where mDNS/DHCP is blocked, pin both ends:

- **Pi:** `sudo nmcli con add type ethernet ifname eth0 con-name demo-link ipv4.method manual ipv4.addresses 10.0.0.1/24`
- **Windows:** Settings → Network & Internet → Ethernet → *Edit* IP assignment →
  Manual → IPv4 on → IP `10.0.0.2`, mask `255.255.255.0`, gateway blank.
- Then use `http://10.0.0.1:8000/` everywhere below instead of the `.local` name.

> Tip: keep the Pi reachable over Wi-Fi/SSH while the Ethernet cable is reserved
> for the demo stream.

## 2. Install on the Pi

```bash
# uv handles the Python 3.12 toolchain + deps (see https://docs.astral.sh/uv/)
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone <this-repo> rp-live-demo && cd rp-live-demo
uv sync
```

Copy your model over (and a labels file if you have one):

```bash
scp model.onnx labels.txt pi@<pi-wifi-ip>:~/rp-live-demo/
```

`labels.txt` is one class name per line, in the model's output-index order. If
omitted, the overlay shows `class <index>` instead of names.

## 3. Run the demo

On the Pi:

```bash
uv run python main.py --model model.onnx --labels labels.txt --threads 4
```

Then on the laptop open **`http://raspberrypi.local:8000/`** (or the Pi's IP). To
embed in slides (PowerPoint/Keynote web view, an `<iframe>`, or an OBS *Browser
Source*), point it at that same URL. `…:8000/stream` is the raw MJPEG stream;
`…:8000/healthz` reports liveness + current FPS.

### Useful flags

| Flag | Default | Notes |
|------|---------|-------|
| `--model` | (required) | Path to the `.onnx` classifier |
| `--labels` | – | Newline-separated class names |
| `--camera` | `0` | Webcam index or `/dev/videoN` |
| `--cam-width/--cam-height` | `1280×720` | Capture resolution |
| `--input-size` | auto | Override if model input isn't auto-detected |
| `--no-normalize` | off | Skip ImageNet mean/std normalisation |
| `--topk` | `5` | Number of predictions in the overlay |
| `--threads` | `4` | onnxruntime threads (Pi 5 = 4 cores) |
| `--port` | `8000` | HTTP port |
| `--jpeg-quality` | `80` | Stream quality vs. bandwidth |

## 4. Preprocessing assumptions

Defaults match standard ImageNet ViT checkpoints: resize to the model's input
size, BGR→RGB, scale to `[0,1]`, normalise with ImageNet mean/std, `NCHW`. Input
size and layout (`NCHW`/`NHWC`) are auto-detected from the ONNX graph. If your
model expects raw `[0,255]` or no normalisation, pass `--no-normalize`; if your
labels look scrambled, your normalisation or label order is the usual culprit.

## 5. Running it as a service (optional, recommended for a talk)

So the demo survives an accidental Ctrl-C or reboot, install it as a `systemd`
service on the Pi. Create `/etc/systemd/system/vit-demo.service`:

```ini
[Unit]
Description=ViT live demo stream
After=network-online.target

[Service]
WorkingDirectory=/home/pi/rp-live-demo
ExecStart=/home/pi/.local/bin/uv run python main.py --model model.onnx --labels labels.txt
Restart=always
User=pi

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable --now vit-demo
```

## Architecture notes

- A single **inference worker thread** owns the camera and model and publishes
  the latest annotated JPEG to a `FrameBroker`. HTTP clients read from the broker
  independently, so inference FPS is decoupled from however many viewers connect.
- Served by **waitress** (a real WSGI server); Flask's dev server is single
  threaded and would stall the stream.
- ViT on a Pi 5 CPU is the bottleneck. If FPS is too low, shrink the model input
  (`--input-size`), lower `--cam-width/height`, or export/quantise the model to
  int8. The capture and stream are cheap by comparison.
```

# Unicorn


## setup

* add the `libunicorn.so` to `app/unicorn` directory.

## create virtual env
`uv venv`


## activate venev
`source .venv/bin/activate`

## run the app
`uv run fastapi dev`

## give permission to the serial port
`sudo chmod 666 /dev/rfcomm0`

or

`sudo usermod -a -G dialout $USER`

or

Create a file `/etc/udev/rules.d/99-serial.rules` with the following content:
```
KERNEL=="rfcomm[0-9]*", GROUP="dialout", MODE="0666"
```
Then reload the udev rules:
```
sudo udevadm control --reload-rules && sudo udevadm trigger
```

## bluetooth
*   `bluettothctl`
*   `scan on`
*   `pair <mac>`
*   `sudo rfcomm release /dev/rfcomm0`
*   `sudo rfcomm connect /dev/rfcomm0 84:2E:14:09:EC:73`

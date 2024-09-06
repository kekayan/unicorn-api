# Unicorn 


## setup

* add the `libunicorn.so` to `app/unicorn` directory.

## create virtual env
`uv venv`


## activate venev
`source .venv/bin/activate`

## run the app
`uv run fastapi dev`


## bluetooth
*   `bluettothctl`
*   `scan on`
*   `pair <mac>`
*   `sudo rfcomm release /dev/rfcomm0`
*   `sudo rfcomm connect /dev/rfcomm0 84:2E:14:09:EC:73`


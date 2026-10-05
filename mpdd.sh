#!/usr/bin/env bash

# Set the correct port for mpc
export MPD_PORT=8600

while true; do
    # Wait for the player state to change
    mpc idle player > /dev/null

    # Fetch the track name
    SONG=$(mpc current)

    # Push the notification to Android if a song is playing
    if [ ! -z "$SONG" ]; then
        termux-notification --title "Now Playing" --content "$SONG" --id "mpd-status"
    fi
    sleep 1
done


#!/bin/zsh
# SUNUM DUZENI: slaytlar ortada, AVAL sag kenarda telefon gibi, Trap sol kenarda.
cd "$(dirname "$0")"
# ekran boyutu (mantiksal)
B=$(osascript -e 'tell application "Finder" to get bounds of window of desktop')
W=$(echo $B | awk -F", " '{print $3}')
# AVAL — sag kenar, cercevesiz app penceresi (520px genis -> telefon ceperi gorunur)
open -na "Google Chrome" --args --user-data-dir=/tmp/sunum-aval \
  --app="file://$PWD/app/AVAL.html" --window-size=520,940 --window-position=$((W-540)),40
sleep 1
# IMBALANCE TRAP — sol kenar
open -na "Google Chrome" --args --user-data-dir=/tmp/sunum-trap \
  --app="file://$PWD/app/imbalance_trap_mobile.html" --window-size=520,940 --window-position=20,40
sleep 1
# Slaytlar
open "$HOME/Downloads/BA_Degree (1).pptx"

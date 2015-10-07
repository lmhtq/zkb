#!/bin/bash
echo "Clean images link cache in images"
rm -rf images/*
echo "Clean databases of features"
rm -rf database/*
echo "Clean cache databases"
rm -rf cache/*

# Databases

relational and non-realtional databses for storing for ML models.

uses both SQL and Mongodb

install sql
> sudo apt-get install mysql-server
> mysql -uroot -p
(can also be run in container)

import sql dump
> echo "CREATE DATABASE hbtn_0d_tvshows;" | mysql -uroot -p
> curl "https://s3.amazonaws.com/intranet-projects-files/holbertonschool-higher-level_programming+/274/hbtn_0d_tvshows.sql" -s | mysql -uroot -p hbtn_0d_tvshows
> echo "SELECT * FROM tv_genres" | mysql -uroot -p hbtn_0d_tvshows



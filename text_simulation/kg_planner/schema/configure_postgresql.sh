#!/usr/bin/env bash

sudo service postgresql start

# Set a fixed password for CI
if [[ -n "$CI" ]]; then
  ENTERED_PASSWORD="nopass"
else
  ENTERED_PASSWORD="$1"
fi

while true; do
  if [[ -z "${ENTERED_PASSWORD// }" ]]; then
    echo "Enter a password that will be used for local connections to the database. The password should not be empty or only whitespaces. This will be stored in plaintext in the current user account."
    read -sp 'Password: ' ENTERED_PASSWORD
  else
    break
  fi
done

schema_path="text_simulation/kg_planner/schema/schema_postgresql.sql"

if [ -n "$2" ]; then
  DB_NAME="$2"
else
  DB_NAME="knowledge_base"
fi

sudo -u postgres dropdb $DB_NAME --if-exists
sudo -u postgres createdb $DB_NAME
sudo -u postgres psql -d $DB_NAME -f $schema_path
sudo -u postgres psql -c "ALTER USER postgres WITH PASSWORD '$ENTERED_PASSWORD'"

# Store the password in a dotfile so we can avoid authentication elsewhere
cat > ~/.pgpass <<EOF
# hostname:port:database:username:password
localhost:*:$DB_NAME:postgres:$ENTERED_PASSWORD
EOF

chmod 600 ~/.pgpass

if [[ -n "$CI" ]]; then
  echo "Executing Industrial CI setup"
  # The previous invocations won't work because the postgres user doesn't
  # have permission to access the working directory
  POSTGRES_HOME=/var/lib/postgresql
  cp sql/schema_postgresql.sql $POSTGRES_HOME
  sudo -u postgres psql -d $DB_NAME -f $POSTGRES_HOME/schema_postgresql.sql
fi
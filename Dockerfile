FROM alpine:latest

EXPOSE 8080

WORKDIR /TwitchChatAtAGlance
COPY . /TwitchChatAtAGlance
RUN apk update && apk upgrade
RUN apk add --no-cache python3-dev py3-numpy py3-scikit-learn py3-mysqlclient
RUN apk add --update --no-cache python3 && ln -sf /usr/bin/python
RUN python3 -m ensurepip
RUN pip3 install --no-cache --upgrade pip setuptools
RUN pip3 install --no-cache-dir -r requirements.txt

RUN chmod +x aSwap.sh

#CMD ["python3","TwitchChatAtAGlance.py"] #Use this if running locally
CMD ./aSwap.sh #We run this first because we are running on less than required RAM
    def process(self):
        event = self.receive_message()
        event.set_default_value()

        csvfile = io.StringIO()
        writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames,
                                quoting=csv.QUOTE_MINIMAL, delimiter=str(";"),
                                extrasaction='ignore', lineterminator='\n')
        writer.writeheader()
        writer.writerow(event)
        attachment = csvfile.getvalue()

        if self.http_verify_cert and self.smtp_class == smtplib.SMTP_SSL:
            kwargs = {'context': ssl.create_default_context()}
        else:
            kwargs = {}

        with self.smtp_class(self.parameters.smtp_host, self.parameters.smtp_port,
                             **kwargs) as smtp:
            if self.starttls:
                if self.http_verify_cert:
                    smtp.starttls(context=ssl.create_default_context())
                else:
                    smtp.starttls()
            if self.username and self.password:
                smtp.auth(smtp.auth_plain, user=self.username, password=self.password)
            msg = MIMEMultipart()
            if self.parameters.text:
                msg.attach(MIMEText(self.parameters.text.format(ev=event)))
            msg.attach(MIMEText(attachment, 'csv'))
            msg['Subject'] = self.parameters.subject.format(ev=event)
            msg['From'] = self.parameters.mail_from.format(ev=event)
            msg['To'] = self.parameters.mail_to.format(ev=event)
            smtp.send_message(msg, from_addr=self.parameters.mail_from,
                              to_addrs=self.parameters.mail_to.format(ev=event))

        self.acknowledge_message()
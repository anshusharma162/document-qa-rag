# Northwind Analytics – Information Security Policy

Version 2.0 · Owner: IT & Security · Mandatory for employees, contractors and interns

## 1. Passwords and authentication

Passwords must be at least 14 characters long and must not reuse any of the last 5 passwords. Passphrases made of several unrelated words are recommended.

Multi-factor authentication (MFA) is mandatory for email, the VPN, the HR portal, cloud consoles and source-code repositories. Approved authenticators are the company authenticator app and hardware security keys. SMS codes are not accepted for administrator accounts.

Passwords must be stored only in the company-approved password manager. Sharing passwords over chat or email is a policy violation.

Accounts are locked after 6 consecutive failed login attempts and unlock automatically after 30 minutes, or immediately through the IT help desk.

## 2. Devices

Company laptops must have full-disk encryption enabled and the endpoint protection agent running at all times. Operating system security updates must be installed within 7 days of release; critical updates within 48 hours.

Screens must lock automatically after 5 minutes of inactivity. Employees must lock their screen whenever they leave their desk.

Personal devices may access company email and chat only through the managed mobile app. Company data must not be downloaded to personal laptops.

Lost or stolen devices must be reported to the IT help desk within 2 hours of discovery so that the device can be remotely wiped.

## 3. Data classification

Data is classified into four levels:

- **Public** – approved for external release, such as marketing material.
- **Internal** – general business information, such as internal wikis.
- **Confidential** – customer data, financial results before release, contracts.
- **Restricted** – personal data of employees and customers, credentials, encryption keys.

Confidential and Restricted data must not be uploaded to unapproved cloud services, including personal file-sharing accounts and public AI chat tools. Restricted data may only be shared externally with a signed data processing agreement and encryption in transit.

## 4. Email and phishing

Employees must report suspected phishing emails using the "Report Phish" button in the email client. Do not forward suspicious emails to colleagues.

Any request received by email to change a vendor's bank details must be verified by a phone call to a number already on file, never a number given in the email.

## 5. Access management

Access follows the principle of least privilege. Managers review their team's access every quarter. Access for leavers is removed on their last working day by 18:00 IST.

Production databases can only be accessed through the bastion host, and every session is logged.

## 6. Incident reporting

Any suspected security incident, such as malware, a data leak or unauthorised access, must be reported to security@northwind.example within 1 hour of discovery. The security team classifies incidents as P1 to P4; P1 incidents trigger the incident response team immediately, 24×7.

Where personal data is affected, the Data Protection Officer decides whether regulators and affected individuals must be notified.

## 7. Training

All staff complete security awareness training within 30 days of joining and annually after that. Employees who fail two phishing simulations in a quarter must complete a refresher module.

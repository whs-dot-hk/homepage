+++
title = "Get started with the AWS Local Zone in Istanbul"
description = "Step-by-step walkthrough from opting in to the Istanbul Local Zone (eu-central-1-ist-1a), creating a VPC with one Local Zone subnet, to launching a Debian EC2 instance."
date = 2026-11-28T07:50:00Z
updated = 2026-11-28T08:10:00Z
template = "article.html"

[taxonomies]
collections = ["Article"]

[extra]
category = "Technology"
category_slug = "technology"
poster = "/media/aws-local-zone-istanbul/10-instance-running.png"
+++

AWS Local Zones bring compute, storage, and networking closer to end users for single-digit millisecond latency. The **Istanbul Local Zone** (`eu-central-1-ist-1a`, Zone ID `euc1-ist1-az1`) extends the **Europe (Frankfurt)** (`eu-central-1`) parent region.

This walkthrough covers the full end-to-end flow in the AWS Management Console (click any screenshot to zoom in):

1. Opting in to the Istanbul Local Zone via AWS Global Views
2. Creating a VPC with **one** subnet in `eu-central-1-ist-1a`
3. Launching a **Debian** EC2 virtual machine (`c7i.large`) in that Local Zone subnet

## 1. Opt in to the Istanbul Local Zone

Before you can create subnets or launch instances in a Local Zone, your AWS account must opt in to the zone.

1. Switch your AWS Console region to **Europe (Frankfurt)** (`eu-central-1`).
2. Open the **EC2 Console** and navigate to **Settings** → **Zones**. AWS now manages Local Zone opt-ins in **AWS Global Views**—click **Open Global Views**.

![EC2 Settings Zones tab showing the Open Global Views button](/media/aws-local-zone-istanbul/01-ec2-settings-zones.png)

3. In **AWS Global View → Regions and Zones**, switch to the **Local Zones** tab and locate **`eu-central-1-ist-1a`** (`Turkey (Istanbul)`, Parent Region `Europe (Frankfurt)`, Zone ID `euc1-ist1-az1`).
4. Select the radio button next to `eu-central-1-ist-1a` and click **Opt-in** (if its status is `Not opted in`). Once enabled, the status updates to **Opted in**.

![AWS Global View Local Zones tab showing eu-central-1-ist-1a Opted in](/media/aws-local-zone-istanbul/02-local-zones-istanbul.png)

## 2. Create a VPC with one Istanbul Local Zone subnet

Next, create a VPC in `eu-central-1` with a single public subnet placed in `eu-central-1-ist-1a`.

1. Open the **VPC Console** (`eu-central-1`) and click **Create VPC**.
2. Choose **VPC and more**, set a name prefix (for example `istanbul-lz`), and keep the IPv4 CIDR block (`10.0.0.0/16`).

![Create VPC wizard with VPC and more selected](/media/aws-local-zone-istanbul/03-create-vpc-wizard.png)

3. Under **Number of Availability Zones (AZs)**, select **1**. Expand **Customize AZs**, open the **First availability zone** dropdown, and select **`euc1-ist1-az1 (eu-central-1-ist-1a)`**. Set **Number of public subnets** to **1**, **Number of private subnets** to **0**, and **NAT gateways** to **None**, then click **Create VPC**.

![Create VPC wizard selecting 1 AZ and choosing eu-central-1-ist-1a from the dropdown](/media/aws-local-zone-istanbul/04-create-vpc-az-dropdown.png)

4. If you are adding a Local Zone subnet to an existing VPC (via **VPC → Subnets → Create subnet**), select your VPC (`istanbul-lz-vpc`), enter a subnet name (`istanbul-lz-subnet-public-ist-1a`), open the **Availability Zone** dropdown, and choose **`Turkey (Istanbul) / euc1-ist1-az1 (eu-central-1-ist-1a)`** with CIDR `10.0.16.0/20`.

![Create subnet Availability Zone dropdown showing Turkey (Istanbul) / eu-central-1-ist-1a](/media/aws-local-zone-istanbul/05-create-subnet-az-dropdown.png)

5. Click **Create subnet** and verify the subnet details: **Availability Zone** is `euc1-ist1-az1 (eu-central-1-ist-1a)`, **Network border group** is `eu-central-1-ist-1`, and **State** is **Available**.

![Subnet details confirming eu-central-1-ist-1a and Available state](/media/aws-local-zone-istanbul/06-subnet-created.png)

## 3. Launch a Debian EC2 VM in the Istanbul Local Zone

Now launch a Debian EC2 instance inside the Istanbul Local Zone subnet.

1. Open **EC2 → Instances → Launch an instance**.
2. Enter a name (for example `istanbul-lz-vm`) and select **Debian** under **Application and OS Images (Quick Start)** (`Debian 13 (HVM), SSD Volume Type`).

![Launch an instance with Debian selected](/media/aws-local-zone-istanbul/07-launch-instance-debian.png)

3. Under **Instance type**, choose an instance family supported in the Istanbul Local Zone—such as **`c7i.large`** (2 vCPU, 4 GiB memory)—and select your **Key pair (login)**.

![Instance type c7i.large and key pair selection](/media/aws-local-zone-istanbul/08-launch-instance-type-keypair.png)

4. Expand **Network settings → Edit** and configure:
   - **VPC**: `istanbul-lz-vpc`
   - **Subnet**: `istanbul-lz-subnet-public-ist-1a` (`eu-central-1-ist-1a`)
   - **Auto-assign public IP**: **Enable**
   - **Firewall (security groups)**: Allow SSH (port 22) as needed.

![Network settings configured with istanbul-lz-vpc and Istanbul Local Zone subnet](/media/aws-local-zone-istanbul/09-launch-instance-network.png)

5. Click **Launch instance**, then open the instance summary page. Confirm **Instance state** is **Running**, **Instance type** is **`c7i.large`**, and **Subnet ID** points to `istanbul-lz-subnet-public-ist-1a`. When connecting over SSH to a Debian AMI, use the default username **`admin`** (`ssh admin@<public-ip>`).

![Running EC2 instance summary in Istanbul Local Zone](/media/aws-local-zone-istanbul/10-instance-running.png)

## 4. Walkthrough video

<video controls preload="metadata" style="width: 100%; border-radius: 12px;" poster="/media/aws-local-zone-istanbul/10-instance-running.png">
  <source src="/media/aws-local-zone-istanbul/walkthrough.mp4" type="video/mp4">
  <track kind="subtitles" srclang="en" label="English" src="/media/aws-local-zone-istanbul/walkthrough.vtt">
</video>

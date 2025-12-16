/**
 * @file nic_napi_driver.c
 * @brief NAPI-based NIC Driver with Interrupt Mitigation
 * 
 * Features:
 * - NAPI polling for RX
 * - Interrupt mitigation
 * - Budget-based packet processing
 * - High-performance packet handling
 */

 #include "../common/pcie_common.h"

 #define DRIVER_NAME "nic_napi"
 #define DRIVER_VERSION "1.0"
 
 MODULE_LICENSE("GPL");
 MODULE_AUTHOR("Abhi");
 MODULE_DESCRIPTION("NAPI-based NIC Driver");
 MODULE_VERSION(DRIVER_VERSION);
 
 #define NAPI_POLL_WEIGHT 64
 
 struct nic_napi_private {
     struct pci_dev *pdev;
     struct net_device *netdev;
     void __iomem *bar0;
     
     /* NAPI */
     struct napi_struct napi;
     
     /* DMA rings */
     struct rx_descriptor *rx_ring;
     dma_addr_t rx_ring_dma;
     
     struct tx_descriptor *tx_ring;
     dma_addr_t tx_ring_dma;
     
     /* RX buffers */
     struct sk_buff **rx_skb;
     dma_addr_t *rx_dma;
     
     /* TX buffers */
     struct sk_buff **tx_skb;
     dma_addr_t *tx_dma;
     
     u32 tx_head;
     u32 tx_tail;
     u32 rx_head;
     u32 rx_tail;
     
     /* Stats */
     u64 napi_polls;
     u64 napi_complete;
     u64 interrupts;
     
     spinlock_t tx_lock;
 };
 
 /* Enable NAPI and disable interrupts */
 static void nic_napi_enable(struct nic_napi_private *priv)
 {
     napi_enable(&priv->napi);
     
     /* Disable interrupts (will be enabled by NAPI complete) */
     iowrite32(0x0, priv->bar0 + 0x04); /* Interrupt mask register */
 }
 
 /* Disable NAPI */
 static void nic_napi_disable(struct nic_napi_private *priv)
 {
     napi_disable(&priv->napi);
 }
 
 /* Process received packets */
 static int nic_clean_rx(struct nic_napi_private *priv, int budget)
 {
     int cleaned = 0;
     
     while (cleaned < budget) {
         struct rx_descriptor *desc = &priv->rx_ring[priv->rx_tail];
         struct sk_buff *skb;
         u16 length;
         
         /* Check if descriptor is done */
         if (!(desc->status & 0x1))
             break;
         
         /* Get length */
         length = desc->length;
         
         /* Get skb */
         skb = priv->rx_skb[priv->rx_tail];
         if (!skb)
             break;
         
         /* Unmap DMA */
         dma_unmap_single(&priv->pdev->dev, priv->rx_dma[priv->rx_tail],
                         MAX_PACKET_SIZE, DMA_FROM_DEVICE);
         
         /* Setup skb */
         skb_put(skb, length);
         skb->protocol = eth_type_trans(skb, priv->netdev);
         skb->ip_summed = CHECKSUM_UNNECESSARY;
         
         /* Pass to stack using NAPI */
         napi_gro_receive(&priv->napi, skb);
         
         /* Allocate new buffer */
         skb = netdev_alloc_skb(priv->netdev, MAX_PACKET_SIZE);
         if (skb) {
             priv->rx_skb[priv->rx_tail] = skb;
             priv->rx_dma[priv->rx_tail] = dma_map_single(&priv->pdev->dev,
                                                          skb->data,
                                                          MAX_PACKET_SIZE,
                                                          DMA_FROM_DEVICE);
             desc->buffer_addr = priv->rx_dma[priv->rx_tail];
             desc->length = MAX_PACKET_SIZE;
         }
         
         /* Clear descriptor status */
         desc->status = 0;
         
         /* Advance tail */
         priv->rx_tail = (priv->rx_tail + 1) % RX_RING_SIZE;
         cleaned++;
     }
     
     return cleaned;
 }
 
 /* NAPI poll function */
 static int nic_napi_poll(struct napi_struct *napi, int budget)
 {
     struct nic_napi_private *priv = container_of(napi,
                                                   struct nic_napi_private,
                                                   napi);
     int work_done;
     
     priv->napi_polls++;
     
     /* Process RX packets */
     work_done = nic_clean_rx(priv, budget);
     
     pcie_dbg("NAPI poll: budget=%d work_done=%d", budget, work_done);
     
     /* If we processed less than budget, we're done */
     if (work_done < budget) {
         napi_complete_done(napi, work_done);
         
         /* Re-enable interrupts */
         iowrite32(0x1, priv->bar0 + 0x04);
         
         priv->napi_complete++;
     }
     
     return work_done;
 }
 
 /* Interrupt handler */
 static irqreturn_t nic_napi_interrupt(int irq, void *dev_id)
 {
     struct net_device *netdev = dev_id;
     struct nic_napi_private *priv = netdev_priv(netdev);
     u32 status;
     
     /* Read interrupt status */
     status = ioread32(priv->bar0 + 0x00);
     
     if (!(status & 0x1))
         return IRQ_NONE;
     
     priv->interrupts++;
     
     /* Disable interrupts */
     iowrite32(0x0, priv->bar0 + 0x04);
     
     /* Clear interrupt */
     iowrite32(0x1, priv->bar0 + 0x00);
     
     /* Schedule NAPI */
     if (napi_schedule_prep(&priv->napi)) {
         __napi_schedule(&priv->napi);
     }
     
     return IRQ_HANDLED;
 }
 
 /* Allocate RX buffers */
 static int nic_alloc_rx_buffers(struct nic_napi_private *priv)
 {
     int i;
     
     for (i = 0; i < RX_RING_SIZE; i++) {
         struct sk_buff *skb = netdev_alloc_skb(priv->netdev, MAX_PACKET_SIZE);
         if (!skb)
             return -ENOMEM;
         
         priv->rx_skb[i] = skb;
         
         priv->rx_dma[i] = dma_map_single(&priv->pdev->dev,
                                          skb->data,
                                          MAX_PACKET_SIZE,
                                          DMA_FROM_DEVICE);
         
         if (dma_mapping_error(&priv->pdev->dev, priv->rx_dma[i])) {
             dev_kfree_skb(skb);
             return -ENOMEM;
         }
         
         /* Setup descriptor */
         priv->rx_ring[i].buffer_addr = priv->rx_dma[i];
         priv->rx_ring[i].length = MAX_PACKET_SIZE;
         priv->rx_ring[i].status = 0;
     }
     
     return 0;
 }
 
 /* Free RX buffers */
 static void nic_free_rx_buffers(struct nic_napi_private *priv)
 {
     int i;
     
     for (i = 0; i < RX_RING_SIZE; i++) {
         if (priv->rx_skb[i]) {
             dma_unmap_single(&priv->pdev->dev, priv->rx_dma[i],
                             MAX_PACKET_SIZE, DMA_FROM_DEVICE);
             dev_kfree_skb(priv->rx_skb[i]);
             priv->rx_skb[i] = NULL;
         }
     }
 }
 
 /* Network device operations */
 static int nic_napi_open(struct net_device *netdev)
 {
     struct nic_napi_private *priv = netdev_priv(netdev);
     int ret;
     
     pcie_info("Opening NAPI network interface");
     
     /* Allocate RX buffers */
     ret = nic_alloc_rx_buffers(priv);
     if (ret) {
         pcie_err("Failed to allocate RX buffers");
         return ret;
     }
     
     /* Request IRQ */
     ret = request_irq(priv->pdev->irq, nic_napi_interrupt,
                      IRQF_SHARED, DRIVER_NAME, netdev);
     if (ret) {
         pcie_err("Failed to request IRQ");
         nic_free_rx_buffers(priv);
         return ret;
     }
     
     /* Enable NAPI */
     nic_napi_enable(priv);
     
     /* Enable interrupts */
     iowrite32(0x1, priv->bar0 + 0x04);
     
     /* Start TX queue */
     netif_start_queue(netdev);
     
     pcie_info("NAPI interface opened");
     return 0;
 }
 
 static int nic_napi_stop(struct net_device *netdev)
 {
     struct nic_napi_private *priv = netdev_priv(netdev);
     
     pcie_info("Stopping NAPI network interface");
     
     /* Stop TX queue */
     netif_stop_queue(netdev);
     
     /* Disable interrupts */
     iowrite32(0x0, priv->bar0 + 0x04);
     
     /* Disable NAPI */
     nic_napi_disable(priv);
     
     /* Free IRQ */
     free_irq(priv->pdev->irq, netdev);
     
     /* Free RX buffers */
     nic_free_rx_buffers(priv);
     
     pcie_info("NAPI interface stopped");
     pcie_info("Stats: polls=%llu complete=%llu interrupts=%llu",
              priv->napi_polls, priv->napi_complete, priv->interrupts);
     
     return 0;
 }
 
 static netdev_tx_t nic_napi_xmit(struct sk_buff *skb,
                                  struct net_device *netdev)
 {
     struct nic_napi_private *priv = netdev_priv(netdev);
     struct tx_descriptor *desc;
     unsigned long flags;
     u32 next_head;
     
     spin_lock_irqsave(&priv->tx_lock, flags);
     
     next_head = (priv->tx_head + 1) % TX_RING_SIZE;
     
     /* Check if ring is full */
     if (next_head == priv->tx_tail) {
         netif_stop_queue(netdev);
         spin_unlock_irqrestore(&priv->tx_lock, flags);
         return NETDEV_TX_BUSY;
     }
     
     /* Get descriptor */
     desc = &priv->tx_ring[priv->tx_head];
     
     /* Map DMA */
     priv->tx_dma[priv->tx_head] = dma_map_single(&priv->pdev->dev,
                                                   skb->data,
                                                   skb->len,
                                                   DMA_TO_DEVICE);
     
     if (dma_mapping_error(&priv->pdev->dev, priv->tx_dma[priv->tx_head])) {
         spin_unlock_irqrestore(&priv->tx_lock, flags);
         return NETDEV_TX_OK; /* Drop packet */
     }
     
     /* Setup descriptor */
     desc->buffer_addr = priv->tx_dma[priv->tx_head];
     desc->length = skb->len;
     desc->cmd = 0x1; /* Send */
     desc->status = 0;
     
     /* Save skb */
     priv->tx_skb[priv->tx_head] = skb;
     
     /* Advance head */
     priv->tx_head = next_head;
     
     /* Trigger TX (device-specific) */
     iowrite32(priv->tx_head, priv->bar0 + 0x10);
     
     spin_unlock_irqrestore(&priv->tx_lock, flags);
     
     return NETDEV_TX_OK;
 }
 
 static const struct net_device_ops nic_napi_netdev_ops = {
     .ndo_open            = nic_napi_open,
     .ndo_stop            = nic_napi_stop,
     .ndo_start_xmit      = nic_napi_xmit,
     .ndo_validate_addr   = eth_validate_addr,
 };
 
 static int nic_napi_probe(struct pci_dev *pdev,
                           const struct pci_device_id *id)
 {
     struct net_device *netdev;
     struct nic_napi_private *priv;
     int ret;
     u8 mac_addr[ETH_ALEN] = {0x00, 0x22, 0x33, 0x44, 0x55, 0x66};
     
     pcie_info("Probing NAPI NIC device");
     
     /* Allocate network device */
     netdev = alloc_etherdev(sizeof(struct nic_napi_private));
     if (!netdev)
         return -ENOMEM;
     
     priv = netdev_priv(netdev);
     priv->netdev = netdev;
     priv->pdev = pdev;
     pci_set_drvdata(pdev, netdev);
     
     spin_lock_init(&priv->tx_lock);
     
     /* Enable device */
     ret = pci_enable_device(pdev);
     if (ret)
         goto err_free_netdev;
     
     /* Set DMA mask */
     ret = dma_set_mask_and_coherent(&pdev->dev, DMA_BIT_MASK(64));
     if (ret)
         ret = dma_set_mask_and_coherent(&pdev->dev, DMA_BIT_MASK(32));
     if (ret)
         goto err_disable_device;
     
     pci_set_master(pdev);
     
     ret = pci_request_regions(pdev, DRIVER_NAME);
     if (ret)
         goto err_disable_device;
     
     /* Map BAR0 */
     priv->bar0 = pci_iomap(pdev, BAR_0, pci_resource_len(pdev, BAR_0));
     if (!priv->bar0) {
         ret = -ENOMEM;
         goto err_release_regions;
     }
     
     /* Allocate descriptor rings */
     priv->rx_ring = dma_alloc_coherent(&pdev->dev,
                                        RX_RING_SIZE * sizeof(struct rx_descriptor),
                                        &priv->rx_ring_dma,
                                        GFP_KERNEL);
     if (!priv->rx_ring) {
         ret = -ENOMEM;
         goto err_unmap_bar;
     }
     
     priv->tx_ring = dma_alloc_coherent(&pdev->dev,
                                        TX_RING_SIZE * sizeof(struct tx_descriptor),
                                        &priv->tx_ring_dma,
                                        GFP_KERNEL);
     if (!priv->tx_ring) {
         ret = -ENOMEM;
         goto err_free_rx_ring;
     }
     
     /* Allocate buffer arrays */
     priv->rx_skb = kcalloc(RX_RING_SIZE, sizeof(struct sk_buff *), GFP_KERNEL);
     priv->rx_dma = kcalloc(RX_RING_SIZE, sizeof(dma_addr_t), GFP_KERNEL);
     priv->tx_skb = kcalloc(TX_RING_SIZE, sizeof(struct sk_buff *), GFP_KERNEL);
     priv->tx_dma = kcalloc(TX_RING_SIZE, sizeof(dma_addr_t), GFP_KERNEL);
     
     if (!priv->rx_skb || !priv->rx_dma || !priv->tx_skb || !priv->tx_dma) {
         ret = -ENOMEM;
         goto err_free_buffers;
     }
     
     /* Setup network device */
     netdev->netdev_ops = &nic_napi_netdev_ops;
     memcpy(netdev->dev_addr, mac_addr, ETH_ALEN);
     
     /* Initialize NAPI */
     netif_napi_add(netdev, &priv->napi, nic_napi_poll, NAPI_POLL_WEIGHT);
     
     /* Register network device */
     ret = register_netdev(netdev);
     if (ret)
         goto err_napi_del;
     
     pcie_info("NAPI NIC registered: %s", netdev->name);
     pcie_info("MAC: %pM, NAPI weight: %d", netdev->dev_addr, NAPI_POLL_WEIGHT);
     
     return 0;
     
 err_napi_del:
     netif_napi_del(&priv->napi);
 err_free_buffers:
     kfree(priv->tx_dma);
     kfree(priv->tx_skb);
     kfree(priv->rx_dma);
     kfree(priv->rx_skb);
     dma_free_coherent(&pdev->dev,
                      TX_RING_SIZE * sizeof(struct tx_descriptor),
                      priv->tx_ring, priv->tx_ring_dma);
 err_free_rx_ring:
     dma_free_coherent(&pdev->dev,
                      RX_RING_SIZE * sizeof(struct rx_descriptor),
                      priv->rx_ring, priv->rx_ring_dma);
 err_unmap_bar:
     pci_iounmap(pdev, priv->bar0);
 err_release_regions:
     pci_release_regions(pdev);
 err_disable_device:
     pci_disable_device(pdev);
 err_free_netdev:
     free_netdev(netdev);
     return ret;
 }
 
 static void nic_napi_remove(struct pci_dev *pdev)
 {
     struct net_device *netdev = pci_get_drvdata(pdev);
     struct nic_napi_private *priv = netdev_priv(netdev);
     
     pcie_info("Removing NAPI NIC device");
     
     unregister_netdev(netdev);
     netif_napi_del(&priv->napi);
     
     kfree(priv->tx_dma);
     kfree(priv->tx_skb);
     kfree(priv->rx_dma);
     kfree(priv->rx_skb);
     
     dma_free_coherent(&pdev->dev,
                      TX_RING_SIZE * sizeof(struct tx_descriptor),
                      priv->tx_ring, priv->tx_ring_dma);
     dma_free_coherent(&pdev->dev,
                      RX_RING_SIZE * sizeof(struct rx_descriptor),
                      priv->rx_ring, priv->rx_ring_dma);
     
     pci_iounmap(pdev, priv->bar0);
     pci_release_regions(pdev);
     pci_disable_device(pdev);
     free_netdev(netdev);
     
     pcie_info("NAPI NIC device removed");
 }
 
 static const struct pci_device_id nic_napi_id_table[] = {
     { PCI_DEVICE(DEMO_VENDOR_ID, DEMO_DEVICE_ID) },
     { 0, }
 };
 MODULE_DEVICE_TABLE(pci, nic_napi_id_table);
 
 static struct pci_driver nic_napi_driver = {
     .name       = DRIVER_NAME,
     .id_table   = nic_napi_id_table,
     .probe      = nic_napi_probe,
     .remove     = nic_napi_remove,
 };
 
 module_pci_driver(nic_napi_driver);
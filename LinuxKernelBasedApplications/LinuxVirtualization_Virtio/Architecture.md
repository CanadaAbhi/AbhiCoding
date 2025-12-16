Host Linux (KVM host kernel module: VMCB/VMCS mgmt, EPT/NPT, vCPU sched)
   |
QEMU (userspace VMM #1)              QEMU (userspace VMM #2)
   |  virtio-net-pci  --- tap0 ---+       |  virtio-blk-pci --- disk2.qcow2
   |  virtio-toy-pci (custom)     |       |  virtio-toy-pci (custom)
   v                              |       v
+--------+                        |   +--------+
|Guest 1 |                        |   |Guest 2 |
|virtio- |                        |   |virtio- |
| net.ko |                        |   | blk.ko |
|virtio_ |                        |   |virtio_ |
| toy.ko |                        |   | toy.ko |
+--------+                        |   +--------+
                                   v
                          br0 (Linux bridge) ---> Host Driver
                          (tap0 + tap1, host-side              (vhost-net kernel
                           network back-end)                    thread bypasses
                                                                  QEMU userspace)




virtio_lab/
  qemu-device/            # host-side QEMU device model (C, patches into QEMU tree)
    virtio-toy.h
    virtio-toy.c
    virtio-toy-pci.c
  guest-driver/           # guest-side Linux kernel driver
    virtio_toy.h
    virtio_toy.c
  guest-app/
    toy_app.c              # Buffer/submit_job/wait_for_completion API over virtio-toy
  vfio-demo/
    vfio_dma_demo.c         # VFIO passthrough alternative path
  scripts/
    build_qemu.sh
    launch_guest1_net.sh
    launch_guest2_blk.sh
    setup_bridge.sh
    bench_vmexits.sh

















Real vIOMMU (Intel VT-d emulation / virtio-iommu)	Your fake_smmu.c
IOVA allocator per domain	iova.h-based rbtree allocator
Guest IOMMU page tables	your software page table (rbtree IOVA→phys + perms)
IOTLB	your TLB hit/miss cache
DMAR fault reporting	your fault queue
Per-device domain assignment (iommu_platform)	your two fake devices attached to separate domains











Real vIOMMU (Intel VT-d emulation / virtio-iommu)	Your fake_smmu.c
IOVA allocator per domain	iova.h-based rbtree allocator
Guest IOMMU page tables	your software page table (rbtree IOVA→phys + perms)
IOTLB	your TLB hit/miss cache
DMAR fault reporting	your fault queue
Per-device domain assignment (iommu_platform)	your two fake devices attached to separate domains